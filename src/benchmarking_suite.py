from itertools import islice
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Resize

from src.model import load_checkpoint
from src.preprocessing import LidarImgDataset, train_val_split, TRAIN_DIR, CACHE_DIR, TEST_DIR
from src.projections import build_lidar_projections

def main():
    device = torch.device("cuda")
    transforms = nn.Sequential(Resize(size=(512, 512)))

    model = load_checkpoint("best_model.pt", device=device)
    samples = sorted([p for p in TEST_DIR.iterdir() if p.is_dir()])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss().to(device)

    dataset = LidarImgDataset(sample_path=TRAIN_DIR, cache_path=CACHE_DIR, transforms=transforms)
    train_loader, val_loader = train_val_split(dataset, seed=42, num_workers=0, batch_size=1)

    N_INFER = int(0.025 * len(samples))
    N_TRAIN = int(0.1 * len(train_loader))
    N_VAL   = int(0.1 * len(val_loader))
        
    print('PART 1: Validation...', end='\t')
    model.eval()
    with torch.cuda.nvtx.range("PART 1: Validation"):
        losses = []
        with torch.inference_mode():
            for x, y in islice(val_loader, N_VAL):
                losses.append(criterion(model(x.to(device)), y.to(device)).item())
        print(f"L1 loss (over {len(losses)} batches): {np.mean(losses):.5f}")

    torch.cuda.synchronize()

    print("PART 2: Inference...")
    model.eval()
    with torch.cuda.nvtx.range("PART 2: Inference"):
        print(" Running forward pass on example samples")        
        with torch.inference_mode():
            for i, path in enumerate(islice(samples, N_INFER)):
                with torch.cuda.nvtx.range("preprocess"):
                    x = build_lidar_projections(path, device, transforms)
                with torch.cuda.nvtx.range("forward"):
                    model(x.unsqueeze(0))
                
                print('|', end='', flush=True)
    print('\nDone')
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    model.train()
    print('PART 3: Training...')
    with torch.cuda.nvtx.range("PART 3: Training"):
        for i, (x, y) in enumerate(islice(train_loader, N_TRAIN)):
            with torch.cuda.nvtx.range(f"train_step"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                criterion(out, y).backward()
                optimizer.step()
                
            print('|', end='', flush=True)

        print(f" peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")


if __name__ == "__main__":
    main()