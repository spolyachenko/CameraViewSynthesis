import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.io import decode_image
from torchvision.transforms import Resize
from pathlib import Path

from projections import build_lidar_projections
from model import load_checkpoint


def predict(model: nn.Module, sample_path: Path, device: torch.device, transforms: nn.Sequential) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        sample_input = build_lidar_projections(sample_path, device, transforms)
        if sample_input is None:
            raise RuntimeError(f"preprocessing produced no tensor for {sample_path}")

        out: torch.Tensor = model(sample_input.unsqueeze(0))
        return out.squeeze(0).cpu()


def visualize(model: nn.Module, sample_path: Path, device: torch.device, transforms: nn.Sequential) -> None:
    model.eval()
    plt.rcParams["savefig.bbox"] = "tight"

    with torch.no_grad():
        sample_input = build_lidar_projections(sample_path, device, transforms)
        if sample_input is None:
            raise RuntimeError(f"preprocessing produced no tensor for {sample_path}")

        out_img = model(sample_input.unsqueeze(0)).squeeze(0).cpu()

        cam_name = next(iter((sample_path / "target").iterdir())).stem
        trg_img = transforms(decode_image(sample_path / "target" / (cam_name + ".jpg")).float() / 255.0)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(_tensor_to_rgb(trg_img))
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(_tensor_to_rgb(sample_input[4]))
        axes[0, 1].set_title("Depth Map")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(_tensor_to_rgb(sample_input[:3]))
        axes[1, 0].set_title("Sampled Image")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(_tensor_to_rgb(out_img))
        axes[1, 1].set_title("Predicted")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show()


def _tensor_to_rgb(tens: torch.Tensor) -> np.ndarray:
    t = tens.detach().cpu().float()
    if t.dim() == 2:
        t = t.unsqueeze(0)          # (1, H, W)
    if t.size(0) == 1:
        t = t.expand(3, -1, -1)     # grayscale -> RGB
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255.0).astype(np.uint8)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = nn.Sequential(Resize(size=(512, 512)))

    model = load_checkpoint("best_model.pt", device=device)

    train_dir = Path("cv_dataset_small/train")
    for sample_path in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        visualize(model, sample_path, device, transforms)