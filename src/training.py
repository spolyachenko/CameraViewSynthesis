import os
os.environ["TMPDIR"] = "/tmp"
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize

from model import CustomUNet
from src.preprocessing import (
    CACHE_DIR,
    TRAIN_DIR,
    LidarImgDataset,
    preprocess_dataset,
    train_val_split,
)


def train_and_log_model(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    criterion: nn.L1Loss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    num_epochs: int = 50,
    start_epoch: int = 0,
    best_val_loss: float = None,
) -> None:
    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        train_loss = 0.0
        sample_count = 0

        model.train()
        print(f"  [Train] Starting {len(train_loader)} batches...", end="", flush=True)
        
        for i, (input, trg) in enumerate(train_loader):
            input = input.to(device)
            trg = trg.to(device)

            optim.zero_grad()
            output = model(input)
            loss = criterion(output, trg)
            loss.backward()
            optim.step()

            sample_count += input.size(0)
            train_loss += loss.item() * input.size(0)
            
            if (i + 1) % 3 == 0:
                print(".", end="", flush=True)

        train_loss = train_loss / sample_count
        print(f" Done. Train Loss: {train_loss:.5f}")

        val_loss = 0
        sample_count = 0
        model.eval()
        print(f"  [Val] Starting {len(val_loader)} batches...", end="", flush=True)
        
        with torch.no_grad():
            for i, (input, trg) in enumerate(val_loader):
                input = input.to(device)
                trg = trg.to(device)

                output = model(input)
                loss = criterion(output, trg)

                sample_count += input.size(0)
                val_loss += loss.item() * input.size(0)
                
                if (i + 1) % 3 == 0:
                    print(".", end="", flush=True)

        val_loss = val_loss / sample_count
        print(f" Done. Val Loss: {val_loss:.5f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.flush()

        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "arch": {"depth": 4, "input_dim": 6},
            }
            torch.save(checkpoint, "best_model.pt")
            print(f"  [Checkpoint] Saved new best model! (Val Loss improved to {val_loss:.5f})")


def fetch_training_results(model: nn.Module, optim: torch.optim.Optimizer) -> None:
    model_artifacts = torch.load("best_model.pt", weights_only=True)
    model.load_state_dict(model_artifacts['model'])
    optim.load_state_dict(model_artifacts['optimizer'])


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transforms = nn.Sequential(Resize(size=(512, 512)))

    print("Checking cache / preprocessing dataset...")
    preprocess_dataset(transforms=transforms, device=device)
    print("Preprocessing finished. Loading dataset...")

    dataset = LidarImgDataset(sample_path=TRAIN_DIR, cache_path=CACHE_DIR, transforms=transforms)
    
    train_loader, val_loader = train_val_split(dataset, seed=42, num_workers=4)
    print(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    model = CustomUNet(4, input_dim=6).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    loss = nn.L1Loss()

    start_epoch = 0
    best_val_loss = None

    if Path("best_model.pt").exists():
        print("Found checkpoint. Resuming...")
        ckpt = torch.load("best_model.pt", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.5f}")

    log_dir = Path("training_runs") / "scratch_unet_l1"
    writer = SummaryWriter(log_dir=str(log_dir))

    try:
        train_and_log_model(
            model, optimizer, loss, train_loader, val_loader, device,
            writer=writer, 
            num_epochs=30,
            start_epoch=start_epoch,      
            best_val_loss=best_val_loss,  
        )
    finally:
        writer.close()
        

if __name__ == '__main__':
    main()