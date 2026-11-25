
from itertools import islice

import torch
import torch.nn as nn
from torchvision.transforms import Resize

from src.model import load_checkpoint
from src.preprocessing import (
    LidarImgDataset,
    train_val_split,
    TRAIN_DIR,
    CACHE_DIR,
    TEST_DIR,
)
from src.projections import build_lidar_projections


def main():
    device = torch.device("cuda")
    transforms = nn.Sequential(Resize(size=(512, 512)))

    model = load_checkpoint("best_model.pt", device=device)

    samples = sorted(p for p in TEST_DIR.iterdir() if p.is_dir())

    dataset = LidarImgDataset(
        sample_path=TRAIN_DIR,
        cache_path=CACHE_DIR,
        transforms=transforms,
    )

    train_loader, _ = train_val_split(
        dataset,
        seed=42,
        num_workers=0,
        batch_size=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    N_INFER = 10
    N_TRAIN = int(0.1 * len(train_loader))

    model.eval()
    print("Inference...")

    with torch.cuda.nvtx.range("Inference"):
        with torch.inference_mode():
            for i in range(N_INFER):
                path = samples[0]

                with torch.cuda.nvtx.range("preprocess"):
                    x = build_lidar_projections(path, device, transforms)

                with torch.cuda.nvtx.range("forward"):
                    model(x.unsqueeze(0))

                torch.cuda.synchronize()
                print("|", end="", flush=True)

    print("\nInference done")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    model.train()
    print("Training...")

    with torch.cuda.nvtx.range("Training"):
        for x, y in islice(train_loader, N_TRAIN):
            with torch.cuda.nvtx.range("train_step"):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            print("|", end="", flush=True)

    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated() / 2**20
    print(f"\nPeak VRAM: {peak_memory:.0f} MB")


if __name__ == "__main__":
    main()
