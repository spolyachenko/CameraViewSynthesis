from pathlib import Path
import os
os.environ["TMPDIR"] = "/tmp"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Resize

from projections import build_lidar_projections, TS_LIST, CAMERA_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "cv_dataset_small" / "test"
TRAIN_DIR = PROJECT_ROOT / "cv_dataset_small" / "train"
CACHE_DIR = PROJECT_ROOT / "Cached_images"

def preprocess_dataset(
    train_dir: Path = TRAIN_DIR,
    cache_dir: Path = CACHE_DIR,
    transforms: nn.Sequential | None = None,
    device: torch.device | None = None,
    skip_existing: bool = True
) -> int:
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    transforms = transforms or nn.Sequential(Resize(size=(512, 512)))
    cache_dir.mkdir(exist_ok=True)
    n = 0
    for path in train_dir.iterdir():
        cache_sample = cache_dir / path.name
        if skip_existing and cache_sample.exists() and list(cache_sample.glob('trg_*.pt')):
            continue
        build_lidar_projections(path, device, transforms, cache_dir)
        n += 1
    return n


class LidarImgDataset(Dataset):
    def __init__(self, sample_path: Path, cache_path: Path, transforms: nn.Sequential) -> None:
        super().__init__()

        self.src_img_dir = list(sample_path.iterdir())
        self.cache_path = cache_path
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.src_img_dir) * 13

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_dir: Path = self.src_img_dir[index // 13]
        sample_name = sample_dir.name

        sample_cache_dir = self.cache_path.joinpath(sample_name)

        ts = TS_LIST[(index%13) // 6]
        cam_name = CAMERA_NAMES[(index%13) % 6]
        if index%13 == 12:
            cam_name = list(sample_dir.joinpath('target').iterdir())[0].stem

        src_img = torch.load(sample_cache_dir / (cam_name + '_' + ts + '.pt'))
        src_img = self.transforms(src_img)

        trg_img = torch.load(sample_cache_dir / (f'trg_{cam_name}_{ts}.pt'))
        trg_img = self.transforms(trg_img)

        return src_img, trg_img


def train_val_split(
    dataset: LidarImgDataset,
    seed: int | None = None,
    train_sz: float = 0.8,
    batch_size: int = 1,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    perm = torch.randperm(len(dataset.src_img_dir), generator=g)
    n_train = max(1, int(train_sz * len(perm)))

    train_idx = [i for s in perm[:n_train].tolist() for i in range(s * 13, (s + 1) * 13)]
    val_idx = [i for s in perm[n_train:].tolist() for i in range(s * 13, (s + 1) * 13)]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader