from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from torchcodec.decoders import decode_image

TS_LIST = ("t0", "t1", "target")
CAMERA_NAMES = ("front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear")


def build_lidar_projections(
    sample_path: Path,
    device: torch.device,
    transforms: nn.Sequential,
    cache_dir: Path | None = None
) -> None | torch.Tensor:
    cache_sample_path = None

    if cache_dir is not None:
        cache_dir.mkdir(exist_ok=True)
        cache_sample_path = cache_dir / sample_path.name
        cache_sample_path.mkdir(exist_ok=True)

    def load_array(path, key):
        with np.load(path, allow_pickle=False) as data:
            return data[key]

    with torch.cuda.nvtx.range("Load NPZ"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            lidar_path = sample_path / 'input' / 'lidar.npz'
        
            intensity_future = executor.submit(load_array, lidar_path, 'intensity')
            xyz_future = executor.submit(load_array, lidar_path, 'xyz')

            intensity_np = intensity_future.result()
            xyz_np = xyz_future.result()

    with torch.cuda.nvtx.range("H2D transfer"):
        intensity = torch.from_numpy(intensity_np).to(device)
        xyz = torch.from_numpy(xyz_np).to(device)

    colors = torch.zeros_like(xyz)
    ones = torch.ones(size=(xyz.size(dim=0), 1), device=device)
    points = torch.cat([xyz, ones], dim=1)

    with torch.cuda.nvtx.range("Loading metadata"):
        meta: dict = json.loads(sample_path.joinpath('meta.json').read_text())
    trg_cam = meta['target_camera']

    def project_batch(ts: str, cams: list[str]):
        intr = [meta['intrinsics'][c] for c in cams]
        Ks = torch.tensor([[[i['fx'], 0, i['cx']], [0, i['fy'], i['cy']], [0, 0, 1]] for i in intr], device=device)
        c2ws = torch.tensor([meta['poses_c2w'][ts][c] for c in cams], device=device)

        T = torch.linalg.inv(c2ws)                       # (B,4,4)
        P = Ks @ T[:, :3, :]                              # (B,3,4)
        proj = torch.einsum('bij,nj->bni', P, points)    # (B,N,3)

        z = proj[..., 2]
        return z, proj[..., 0] / z, proj[..., 1] / z

    def depth_maps(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, w: int, h: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_visible = (
            (z > 0) &  
            (x >= 0) & (x < w) &
            (y >= 0) & (y < h)
        )
        xi, yi, zi = x[is_visible].long(), y[is_visible].long(), z[is_visible]

        pointcloud_ids = torch.nonzero(is_visible).squeeze(1)
        img_indexes = yi * w + xi

        depth_map = torch.full(size=(h * w,), fill_value=torch.inf, device=device)
        depth_map.scatter_reduce_(0, img_indexes, zi, reduce="amin", include_self=True)

        is_valid_pixel = (depth_map != torch.inf)
        cand_ids = torch.where(
            zi == depth_map[img_indexes], 
            pointcloud_ids, 
            torch.full_like(pointcloud_ids, h * w)
        )
        
        point_ids_per_pixel = torch.full(size=(h * w,), fill_value=h * w, dtype=torch.int64, device=device)
        point_ids_per_pixel.scatter_reduce_(0, img_indexes, cand_ids, reduce="amin", include_self=True)
        
        used_point_ids = point_ids_per_pixel[is_valid_pixel]

        return depth_map, used_point_ids, is_valid_pixel

    def construct(cam: str, depth_map: torch.Tensor, used_point_ids: torch.Tensor, is_valid_pixel: torch.Tensor) -> torch.Tensor:
        w, h = meta['intrinsics'][cam]['width'], meta['intrinsics'][cam]['height']

        reconstructed_img = torch.zeros(size=(h * w, 3), device=device)
        reconstructed_img[is_valid_pixel] = colors[used_point_ids]
        reconstructed_img = reconstructed_img.view(h, w, 3).permute(2, 0, 1)

        depth_map[is_valid_pixel] = 1.0 / (depth_map[is_valid_pixel] + 1e-6)
        depth_map[~is_valid_pixel] = 0.0
        depth_map = depth_map.view(1, h, w)

        intensity_map = torch.zeros(size=(h * w,), device=device)
        intensity_map[is_valid_pixel] = intensity[used_point_ids]
        intensity_map = intensity_map.view(1, h, w).float() / 255.0

        final_tensor = torch.cat([reconstructed_img, is_valid_pixel.view(1, h, w).float(), depth_map, intensity_map], dim=0)
        return transforms(final_tensor)

    precomp: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        for ts in ('t0', 't1'):
            with torch.cuda.nvtx.range(f"Batch projecting {ts}"):
                z, x, y = project_batch(ts, list(CAMERA_NAMES))

            with torch.cuda.nvtx.range(f"Decode images {ts}"):
                futures = {
                    cam: executor.submit(
                        decode_image,
                        sample_path / 'input' / ts / f'{cam}.jpg',
                        mode="RGB"
                    )
                    for cam in CAMERA_NAMES
                }

                images = {
                    cam: futures[cam].result()
                    for cam in CAMERA_NAMES
                }

            for b, cam in enumerate(CAMERA_NAMES):
                intr = meta['intrinsics'][cam]

                with torch.cuda.nvtx.range(f"Depth map {ts}_{cam}"):
                    depth_map, used_point_ids, is_valid_pixel = depth_maps(
                        z[b], x[b], y[b],
                        intr['width'], intr['height']
                    )

                    precomp[(ts, cam)] = (
                        depth_map,
                        used_point_ids,
                        is_valid_pixel
                    )

                with torch.cuda.nvtx.range(f"Color assignment {ts}_{cam}"):
                    original_img = (
                        images[cam]
                        .to(device)
                        .float()
                        .flatten(start_dim=1)
                        .div_(255.0)
                    )

                    colors[used_point_ids] = (
                        original_img[:, is_valid_pixel].permute(1, 0)
                    )

            del images, futures, z, x, y

    with torch.cuda.nvtx.range("Project target"):
        z, x, y = project_batch('target', [trg_cam])

        intr = meta['intrinsics'][trg_cam]

        target_depth = depth_maps(
            z[0], x[0], y[0],
            intr['width'], intr['height']
        )

    for ts in ('t0', 't1', 'target'):
        cams = [trg_cam] if ts == 'target' else CAMERA_NAMES

        for cam in cams:
            if ts == 'target':
                depth_map, used_point_ids, is_valid_pixel = target_depth
            else:
                if cache_sample_path is None:
                    continue

                depth_map, used_point_ids, is_valid_pixel = precomp[(ts, cam)]

            with torch.cuda.nvtx.range(f"Construct {ts}_{cam}"):
                final_tensor = construct(
                    cam,
                    depth_map,
                    used_point_ids,
                    is_valid_pixel
                )

            if cache_sample_path is None: return final_tensor

            with torch.cuda.nvtx.range(f"Save projection {ts}_{cam}"):
                torch.save(
                    final_tensor.cpu(),
                    cache_sample_path / f'{cam}_{ts}.pt'
                )

            if ts == 'target':
                trg_src = sample_path / 'target' / f'{cam}.jpg'
            else:
                trg_src = sample_path / 'input' / ts / f'{cam}.jpg'

            with torch.cuda.nvtx.range(f"Load GT image {ts}_{cam}"):
                trg = transforms(
                    decode_image(trg_src, mode="RGB").float().div_(255.0)
                )

            with torch.cuda.nvtx.range(f"Save GT image {ts}_{cam}"):
                torch.save(
                    trg.cpu(),
                    cache_sample_path / f'trg_{cam}_{ts}.pt'
                )
