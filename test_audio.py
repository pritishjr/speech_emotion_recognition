"""
Evaluate a saved HuBERT audio-emotion checkpoint on processed audio data.
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE
from src.data.audio_testing_musan import build_noisy_dataset
from train_audio import AudioEmotionDataset, HuBERTClassifier, collate_audio_fn


EMOTION_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


def parse_args():
    parser = argparse.ArgumentParser("Evaluate a trained HuBERT audio classifier")

    parser.add_argument("--checkpoint_path", type=str, default="./models/audio_hubert_final.pth", help="Checkpoint to evaluate.")
    parser.add_argument("--audio_dir", type=str, default="./data/processed/audio-test", help="Directory containing .npy waveforms.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--num_classes", type=int, default=8, help="Fallback num_classes if checkpoint has no args.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Fallback classifier dropout if checkpoint has no args.")
    parser.add_argument("--freeze_cnn", action=argparse.BooleanOptionalAction, default=True, help="Fallback freeze_cnn if checkpoint has no args.")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Fallback freeze_layers if checkpoint has no args.")
    parser.add_argument("--save_predictions", type=str, default=None, help="Optional CSV path for per-sample predictions.")
    parser.add_argument("--compare_snr_db", type=float, nargs="+", default=None, help="Run comparison across these SNR dB levels (e.g., 0 5 10 15).")
    parser.add_argument("--clean_audio_dir", type=str, default="./data/processed/audio", help="Clean audio directory used to synthesize noisy sets.")
    parser.add_argument("--noise_dir", type=str, default="./data/raw/MUSAN/noise/sound-bible", help="MUSAN noise directory.")
    parser.add_argument("--noisy_root_dir", type=str, default="./data/processed/audio-test", help="Root directory for generated per-SNR noisy sets.")
    parser.add_argument("--regenerate_noisy", action=argparse.BooleanOptionalAction, default=True, help="Regenerate noisy files before each SNR evaluation.")
    parser.add_argument("--noise_seed", type=int, default=42, help="Seed for deterministic noise selection/mixing.")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate for MUSAN loading.")
    parser.add_argument("--noise_max_files", type=int, default=None, help="Optional cap for noisy set generation.")
    parser.add_argument("--comparison_output_csv", type=str, default="./reports/audio_snr_comparison.csv", help="Where to save SNR comparison metrics CSV.")

    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_args = checkpoint.get("args", {})
    else:
        state_dict = checkpoint
        saved_args = {}

    return checkpoint, state_dict, saved_args


def build_model(saved_args, cli_args):
    model = HuBERTClassifier(
        num_classes=saved_args.get("num_classes", cli_args.num_classes),
        freeze_cnn=saved_args.get("freeze_cnn", cli_args.freeze_cnn),
        freeze_layers=saved_args.get("freeze_layers", cli_args.freeze_layers),
        dropout=saved_args.get("dropout", cli_args.dropout),
    )
    return model


def compute_classwise_metrics(confusion_matrix):
    tp = confusion_matrix.diag().float()
    support = confusion_matrix.sum(dim=1).float()
    pred_total = confusion_matrix.sum(dim=0).float()

    precision = tp / pred_total.clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)
    f1 = (2 * precision * recall) / (precision + recall).clamp_min(1e-12)

    total = support.sum().clamp_min(1.0)
    accuracy = tp.sum() / total
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / total

    return {
        "accuracy": accuracy.item(),
        "macro_f1": macro_f1.item(),
        "weighted_f1": weighted_f1.item(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    amp_enabled = DEVICE.type == "cuda"

    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    predictions_log = []

    loop = tqdm(loader, desc="Testing", leave=False)
    for batch in loop:
        audio = batch["audio"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        filenames = batch["filename"]

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(audio, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        labels_cpu = labels.cpu()
        preds_cpu = preds.cpu()
        for true_label, pred_label in zip(labels_cpu.tolist(), preds_cpu.tolist()):
            confusion[true_label, pred_label] += 1

        for fname, true_label, pred_label in zip(filenames, labels_cpu.tolist(), preds_cpu.tolist()):
            predictions_log.append((fname, true_label, pred_label))

        running_acc = confusion.diag().sum().item() / max(total_samples, 1)
        loop.set_postfix(
            loss=f"{(total_loss / max(total_samples, 1)):.4f}",
            acc=f"{running_acc:.4f}",
        )

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_classwise_metrics(confusion)

    return avg_loss, metrics, confusion, predictions_log


def maybe_save_predictions(path, predictions_log):
    if not path:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label_id", "true_label_name", "pred_label_id", "pred_label_name"])
        for fname, true_id, pred_id in predictions_log:
            true_name = EMOTION_NAMES[true_id] if true_id < len(EMOTION_NAMES) else str(true_id)
            pred_name = EMOTION_NAMES[pred_id] if pred_id < len(EMOTION_NAMES) else str(pred_id)
            writer.writerow([fname, true_id, true_name, pred_id, pred_name])

    print(f"Saved predictions to: {out_path}")


def maybe_save_snr_comparison(path, comparison_rows):
    if not path:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr_db",
                "loss",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "num_samples",
                "audio_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Saved SNR comparison to: {out_path}")


def format_snr_tag(snr_db):
    return f"{snr_db:g}".replace("-", "m").replace(".", "p")


def build_loader_for_audio_dir(audio_dir, split, batch_size, num_workers):
    dataset = AudioEmotionDataset(
        audio_dir=audio_dir,
        split=split,
        use_augmentation=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return loader


def print_snr_comparison_table(rows):
    print("\n--- SNR COMPARISON ---")
    print(f"{'SNR(dB)':>8} {'Loss':>10} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12} {'Samples':>9}")
    for row in rows:
        print(
            f"{row['snr_db']:>8.1f} "
            f"{row['loss']:>10.4f} "
            f"{row['accuracy']:>10.4f} "
            f"{row['macro_f1']:>10.4f} "
            f"{row['weighted_f1']:>12.4f} "
            f"{row['num_samples']:>9d}"
        )


def print_report(avg_loss, metrics, confusion):
    print("\n--- TEST METRICS ---")
    print(f"Loss       : {avg_loss:.4f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    print("\n--- PER-CLASS METRICS ---")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for class_id in range(confusion.shape[0]):
        class_name = EMOTION_NAMES[class_id] if class_id < len(EMOTION_NAMES) else str(class_id)
        p = metrics["precision"][class_id].item()
        r = metrics["recall"][class_id].item()
        f1 = metrics["f1"][class_id].item()
        s = int(metrics["support"][class_id].item())
        print(f"{class_name:<12} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {s:>10d}")

    print("\n--- CONFUSION MATRIX (rows=true, cols=pred) ---")
    print(confusion)


def main():
    args = parse_args()
    print(f"Evaluating on device: {DEVICE}")

    checkpoint, state_dict, saved_args = load_checkpoint(args.checkpoint_path)
    if isinstance(checkpoint, dict):
        print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")

    model = build_model(saved_args, args).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)

    num_classes = saved_args.get("num_classes", args.num_classes)

    if args.compare_snr_db:
        comparison_rows = []
        for snr_db in args.compare_snr_db:
            snr_tag = format_snr_tag(snr_db)
            snr_audio_dir = Path(args.noisy_root_dir) / f"snr_{snr_tag}db"
            snr_metadata_csv = snr_audio_dir / "metadata.csv"

            print(f"\nPreparing noisy test set for SNR={snr_db:g} dB ...")
            build_noisy_dataset(
                clean_audio_dir=args.clean_audio_dir,
                noise_dir=args.noise_dir,
                output_dir=snr_audio_dir,
                metadata_csv=snr_metadata_csv,
                split=args.split,
                target_sr=args.target_sr,
                snr_db_values=[float(snr_db)],
                seed=args.noise_seed,
                max_files=args.noise_max_files,
                overwrite=args.regenerate_noisy,
                quiet=False,
            )

            loader = build_loader_for_audio_dir(
                audio_dir=str(snr_audio_dir),
                split=args.split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            avg_loss, metrics, confusion, predictions_log = evaluate(
                model=model,
                loader=loader,
                num_classes=num_classes,
            )

            print(f"\nResults for SNR={snr_db:g} dB")
            print_report(avg_loss, metrics, confusion)

            num_samples = int(metrics["support"].sum().item())
            comparison_rows.append(
                {
                    "snr_db": float(snr_db),
                    "loss": float(avg_loss),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "weighted_f1": float(metrics["weighted_f1"]),
                    "num_samples": num_samples,
                    "audio_dir": str(snr_audio_dir),
                }
            )

            if args.save_predictions:
                pred_path = Path(args.save_predictions)
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                per_snr_pred_path = pred_path.with_name(f"{pred_path.stem}_snr_{snr_tag}{pred_path.suffix or '.csv'}")
                maybe_save_predictions(str(per_snr_pred_path), predictions_log)

        print_snr_comparison_table(comparison_rows)
        maybe_save_snr_comparison(args.comparison_output_csv, comparison_rows)
        return

    loader = build_loader_for_audio_dir(
        audio_dir=args.audio_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    avg_loss, metrics, confusion, predictions_log = evaluate(
        model=model,
        loader=loader,
        num_classes=num_classes,
    )

    print_report(avg_loss, metrics, confusion)
    maybe_save_predictions(args.save_predictions, predictions_log)


if __name__ == "__main__":
    main()


