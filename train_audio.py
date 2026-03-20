"""
Audio-only fine-tuning script for emotion classification.

This trains a HuBERT-based classifier on 1D waveform `.npy` files from:
    ./data/processed/audio
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DEVICE
from src.models.audio_model import HuBERTFeatureExtractor

try:
    import wandb
except ImportError:
    wandb = None

#creating the arguments for parsing:
def parse_args():
    parser = argparse.ArgumentParser("Fine-tune HuBERT for audio emotion classification")

    parser.add_argument("--audio_dir", type=str, default="./data/processed/audio", help="Directory containing audio .npy files.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--freeze_cnn", action=argparse.BooleanOptionalAction, default=True, help="Freeze HuBERT CNN feature extractor.")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of lower HuBERT transformer layers to freeze.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout used in classifier head.")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of emotion classes.")

    parser.add_argument("--use_augmentation", action=argparse.BooleanOptionalAction, default=True, help="Apply light waveform augmentation on training split.")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True, help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="audio-emotion-recognition", help="wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional wandb run name.")

    parser.add_argument("--save_path", type=str, default="./models/audio_hubert_final.pth", help="Path to save final checkpoint.")
    parser.add_argument("--best_save_path", type=str, default="./models/audio_hubert_best.pth", help="Path to save best validation checkpoint.")

    return parser.parse_args()

#setting the randomness seed which remains consistent.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#extracting the emotion labels from the filenames.
def parse_label_from_filename(filename):
    parts = filename.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected filename format: {filename}")
    return int(parts[2]) - 1

#extracting the actor ID (label) from the filenames.
def split_by_actor_id(filenames):
    train_set, val_set, test_set = [], [], []
    for fname in filenames:
        try:
            actor_id = int(fname.split("-")[6].split(".")[0])
        except (IndexError, ValueError):
            continue

        if actor_id <= 20:
            train_set.append(fname)
        elif actor_id <= 22:
            val_set.append(fname)
        else:
            test_set.append(fname)
    return train_set, val_set, test_set

#performing "online" augmentation to the audio data.
def audio_augmentation(waveform):
    # Add light Gaussian noise.
    if torch.rand(1).item() > 0.5: #50% probability
        noise_scale = 0.003 * torch.rand(1).item()
        waveform = waveform + torch.randn_like(waveform) * noise_scale

    # Random gain change in dB.
    if torch.rand(1).item() > 0.5:
        gain_db = torch.empty(1).uniform_(-6.0, 6.0).item()
        waveform = waveform * (10 ** (gain_db / 20.0))

    return waveform #returns the 1D .npy waveform.

#initializing the dataset for the train/val/test split.
class AudioEmotionDataset(Dataset):
    def __init__(self, audio_dir, split="train", use_augmentation=True):
        super().__init__()

        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        self.audio_dir = audio_dir
        self.split = split
        self.use_augmentation = use_augmentation and split == "train"

        filenames = [f for f in os.listdir(audio_dir) if f.endswith(".npy")]
        train_set, val_set, test_set = split_by_actor_id(filenames)

        #data split is done based on the actors IDs
        if split == "train":
            self.filenames = sorted(train_set)
        elif split == "val":
            self.filenames = sorted(val_set)
        elif split == "test":
            self.filenames = sorted(test_set)
        else:
            raise ValueError("split must be one of: train, val, test")

        if len(self.filenames) == 0:
            raise ValueError(f"No files found for split='{split}' in {audio_dir}")

    def __len__(self): #len of object is no. of files in the data split.
        return len(self.filenames)

    def __getitem__(self, idx): #index gives the label and audio waveform
        filename = self.filenames[idx]
        audio_path = os.path.join(self.audio_dir, filename)

        audio_np = np.load(audio_path)
        waveform = torch.from_numpy(audio_np).float().flatten()
        if self.use_augmentation:
            waveform = audio_augmentation(waveform)

        label = torch.tensor(parse_label_from_filename(filename), dtype=torch.long)

        return {
            "audio": waveform,
            "label": label,
            "filename": filename,
        }

#collecting the attributes of a sample.
def collate_audio_fn(batch):
    audios = [item["audio"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    filenames = [item["filename"] for item in batch]

    #padding it to get a constant maximum length.
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)

    attention_masks = torch.zeros((len(audios), padded_audios.shape[1]), dtype=torch.long)
    for i, audio in enumerate(audios):
        attention_masks[i, : audio.shape[0]] = 1

    return {
        "audio": padded_audios,
        "attention_mask": attention_masks,
        "label": labels,
        "filename": filenames,
    }

#creatingg the classifier based on HuBERT feature extractor.
class HuBERTClassifier(nn.Module):
    def __init__(self, num_classes=8, freeze_cnn=True, freeze_layers=0, dropout=0.3):
        super().__init__()
        self.audio_backbone = HuBERTFeatureExtractor(
            freeze_cnn=freeze_cnn,
            freeze_layers=freeze_layers,
        )
        
        #mlp layer connecting the last layer of HuBERT.
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(dropout),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes),
        )

    def forward(self, audio_waveforms, attention_mask=None):
        features = self.audio_backbone(audio_waveforms, attention_mask=attention_mask)
        return self.classifier(features)

#calculating metrics for each epoch.
def train_one_epoch(model, loader, criterion, optimizer, scaler, amp_enabled, max_grad_norm):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        audio = batch["audio"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(audio, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{running_loss / (progress.n + 1):.4f}",
            acc=f"{(correct / max(total, 1)):.4f}",
        )

    return running_loss / len(loader), correct / max(total, 1)

#eval on val data per epoch.
@torch.no_grad()
def evaluate(model, loader, criterion, amp_enabled):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Validation", leave=False)
    for batch in progress:
        audio = batch["audio"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(audio, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{running_loss / (progress.n + 1):.4f}",
            acc=f"{(correct / max(total, 1)):.4f}",
        )

    return running_loss / len(loader), correct / max(total, 1)

#training the model over N epochs and saving the model.
#save: best hyyperparameters of the model on val data epoch checkpoint and final trained model (over entirety of the training data).
def train_audio_model(args):
    set_seed(args.seed)
    print(f"Training on device: {DEVICE}")

    train_dataset = AudioEmotionDataset(
        audio_dir=args.audio_dir,
        split="train",
        use_augmentation=args.use_augmentation,
    )
    val_dataset = AudioEmotionDataset(
        audio_dir=args.audio_dir,
        split="val",
        use_augmentation=False,
    )

    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_audio_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_audio_fn,
        pin_memory=pin_memory,
    )

    model = HuBERTClassifier(
        num_classes=args.num_classes,
        freeze_cnn=args.freeze_cnn,
        freeze_layers=args.freeze_layers,
        dropout=args.dropout,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = DEVICE.type == "cuda"
    scaler = GradScaler(device=DEVICE.type, enabled=amp_enabled)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.best_save_path).parent.mkdir(parents=True, exist_ok=True)

    #check if wandb is installed. (run: pip install wandb)
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or run with --no-wandb.")
        run_name = args.wandb_run_name or f"hubert_audio_lr{args.lr}_bs{args.batch_size}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    best_val_acc = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_grad_norm=args.max_grad_norm,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            amp_enabled=amp_enabled,
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if args.wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "args": vars(args),
                },
                args.best_save_path,
            )
            print(f"Saved best checkpoint to {args.best_save_path} (val_acc={val_acc:.4f})")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_acc,
            "args": vars(args),
        },
        args.save_path,
    )
    print(f"Saved final checkpoint to {args.save_path}")

    if args.wandb:
        wandb.finish()


def main():
    args = parse_args()
    train_audio_model(args)


if __name__ == "__main__":
    main()
