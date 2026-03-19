
#training our model here:

#importing the DEVICE form config.py
from config import DEVICE

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.amp import GradScaler

# from .src.models import FusionModel
# from .src.data import MultimodalData, collate_fn

from src import *

from tqdm import tqdm
import wandb

#parsing the arguments
def parse_args():
    
    parser = argparse.ArgumentParser("Training the Fusion Model:")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (float).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch training size.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for our AdamW optimizer.")
    parser.add_argument("--save_path", type=str, default='./models/base_model.pth', help="Saving the model (.pth) in a specific directory.")
    
    parser.add_argument("--audio_dir", type=str, default='./data/processed/audio', help="Directory of audio data.")
    parser.add_argument("--vid_dir", type=str, default='./data/processed/video', help="Directory of video data.")
    
    args = parser.parse_args()
    return args
    
def train(args):
    
    #model is already instantiated.
    print(f"Model is being trained on: {DEVICE}")
    
    #defining the model params:
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    
    #initializing a wandb project:
    wandb.init(
        project="multimodal-emotion-recognition", #name of project dashboard
        name=f"run_lr{lr}_bs{batch_size}", #name of the current run
        config=vars(args) #logging hyper params
    )
    
    #initializing and loading the TRAINING data:
    print("loading training data...")
    training_data = MultimodalData(
        audio_dir=args.audio_dir,
        video_dir=args.vid_dir,
        split='train',
    )
    
    training_data_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    #initializing and loading the VALIDATION data:
    print("loading validation data...")
    val_data = MultimodalData(
        audio_dir=args.audio_dir,
        video_dir=args.vid_dir,
        split='val'
    )
    
    val_data_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True        
    )
    
    #initializing the model:
    model = FusionModel(num_classes=8, use_wo_mask=False)
    model.to(DEVICE)
    
    #loss and optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay = weight_decay
    )
    
    #automatic mixed precision (AMP):
    amp_enabled = DEVICE.type == "cuda" and torch.backends.cudnn.enabled
    scaler = GradScaler(
        device=DEVICE.type,
        enabled=amp_enabled
    )
    
    r"""
        TRAINING STAGE    
    """
    print("beginning training loop...")
    for epoch in range(epochs):
        
        model.train()
        running_loss = 0.0
        training_correct = 0
        training_total = 0
        
        #applying the progress bar tqdm:
        loop = tqdm(
            training_data_loader,
            desc= f"Epoch [{(epoch+1)}/{epochs}]:",
        )
        
        for batch in loop:
            
            #sending data to gpu for computation
            audio_data = batch['audio'].to(DEVICE, non_blocking = True)
            vid_data = batch['video'].to(DEVICE, non_blocking = True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking = True)
            labels = batch['label'].to(DEVICE, non_blocking = True)
            
            optimizer.zero_grad(set_to_none=True)
            
            #converting the data to 16 float point.
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=amp_enabled
            ):
                # Update your model's forward pass to accept the HuBERT attention mask
                logits = model(audio_data, vid_data, audio_attention_mask=attention_mask)
                loss = criterion(logits, labels)
        
            scaler.scale(loss).backward()
            
            #gradient clipping:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer Step & Update Scaler
            scaler.step(optimizer)
            scaler.update()

            # Tracking metrics
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            
            training_total += batch_total
            training_correct += batch_correct

            # Updating tqdm
            current_loss = running_loss / (loop.n + 1)
            current_acc = training_correct / training_total
            loop.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}") 
             
        train_epoch_loss = running_loss / len(training_data_loader)
        train_epoch_acc = training_correct / training_total
        
        print(f"""
              --- TRAINING METRICS ---
              Epoch [{epoch+1}/{args.epochs}] Final
              Loss: {train_epoch_loss:.4f}
              Accuracy: {train_epoch_acc:.4f}
              """)
        
        print("Training done. Evaluation process beginning.")
        
        r"""
            EVALUATOIN STAGE
        """
        model.eval()
        
        val_running_loss, val_correct, val_total = 0.0, 0,0
        
        #tqdm progress bar for val_data
        val_loop = tqdm(val_data_loader, desc=f"Epoch {(epoch+1) / epochs}" ,leave=False)
        
        with torch.no_grad():
            for batch in val_loop:
                audio_waves = batch['audio'].to(DEVICE, non_blocking=True)
                attention_masks = batch['attention_mask'].to(DEVICE, non_blocking=True)
                video_frames = batch['video'].to(DEVICE, non_blocking=True)
                labels = batch['label'].to(DEVICE, non_blocking=True)

                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.float16,
                    enabled=amp_enabled
                ):
                    logits = model(audio_waves, video_frames, audio_attention_mask=attention_masks)
                    loss = criterion(logits, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_loop.set_postfix(loss=f"{(val_running_loss / (val_loop.n + 1)):.4f}")

        val_epoch_loss = val_running_loss / len(val_data_loader)
        val_epoch_acc = val_correct / val_total
        
        print(f"""
              --- VALIDATION METRICS ---
              Epoch : {(epoch + 1) / epochs}
              Loss : {val_epoch_loss:.4f}
              Accuracy : {val_epoch_acc:.4f}
              """)

        print("Validation done.")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_epoch_loss,
            "train/accuracy": train_epoch_acc,
            "val/loss": val_epoch_loss,
            "val/accuracy": val_epoch_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        
    #saving the base_model
    torch.save(model.state_dict(), args.save_path)
    print(f"Phase 1 Complete! Base model saved to '{args.save_path}'.")
    
    pass
def main():
    
    args = parse_args()
    train(args)
    
if __name__ == "__main__":
    main()
