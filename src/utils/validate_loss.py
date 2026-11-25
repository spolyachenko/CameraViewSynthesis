
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Resize

from src.model import load_checkpoint
from src.preprocessing import (
    LidarImgDataset,
    train_val_split,
    TRAIN_DIR,
    CACHE_DIR,
)

def main():
    device = torch.device("cuda")
    transforms = nn.Sequential(Resize(size=(512, 512)))

    model = load_checkpoint("best_model.pt", device=device)
    model.eval()

    dataset = LidarImgDataset(
        sample_path=TRAIN_DIR,
        cache_path=CACHE_DIR,
        transforms=transforms,
    )

    _, val_loader = train_val_split(
        dataset,
        seed=42,
        num_workers=0,
        batch_size=1,
    )

    n_val = int(0.1 * len(val_loader))
    criterion = nn.L1Loss()

    print("Validation...", end="\t")

    losses = []

    with torch.cuda.nvtx.range("Validation"):
        with torch.inference_mode():
            for x, y in islice(val_loader, n_val):
                x = x.to(device)
                y = y.to(device)

                loss = criterion(model(x), y)
                losses.append(loss.item())

    torch.cuda.synchronize()

    print(f"L1 loss: {np.mean(losses):.5f}")
    print(f"Evaluated batches: {len(losses)}")


if __name__ == "__main__":
    main()
