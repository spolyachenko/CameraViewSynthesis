import torch
import torch.nn as nn
import time
import statistics
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torchvision.transforms import Resize

from src.model import CustomUNet, load_checkpoint
from src.preprocessing import LidarImgDataset, train_val_split, TRAIN_DIR, CACHE_DIR
from src.projections import build_lidar_projections

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Benchmarking Suite on {device} ===\n")
    
    transforms = nn.Sequential(Resize(size=(512, 512)))

    print("--- PART 1: Inference (Cold Cache) ---")
    model = load_checkpoint("best_model.pt", device=device)
    model.eval()
    print("Loaded weights from best_model.pt")
    
    all_samples = [p for p in TRAIN_DIR.iterdir() if p.is_dir()][:20]
    
    print("Warming up GPU (5 runs)...")
    for p in all_samples[:5]:
        with torch.inference_mode():
            tensor = build_lidar_projections(p, device, transforms)
            _ = model(tensor.unsqueeze(0))
            
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print("Running 20 inference loops...")
    preproc_times = []
    model_times = []
    
    with torch.inference_mode():
        for p in all_samples:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            tensor = build_lidar_projections(p, device, transforms)
            
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            output = model(tensor.unsqueeze(0))
            
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            
            preproc_times.append((t1 - t0) * 1000)
            model_times.append((t2 - t1) * 1000)

    avg_preproc = statistics.mean(preproc_times)
    avg_model = statistics.mean(model_times)
    total_latency = avg_preproc + avg_model
    preproc_split = (avg_preproc / total_latency) * 100

    print(f"  Preprocessing: {avg_preproc:.2f} ms ({preproc_split:.1f}%)")
    print(f"  Model Forward: {avg_model:.2f} ms ({100-preproc_split:.1f}%)")
    print(f"  Total Latency: {total_latency:.2f} ms (Avg)")
    print(f"  P95 Latency:   {statistics.quantiles([p+m for p,m in zip(preproc_times, model_times)], n=20)[18]:.2f} ms\n")

    print("--- PART 2: Training Loop (Warm Cache) ---")
    dataset = LidarImgDataset(sample_path=TRAIN_DIR, cache_path=CACHE_DIR, transforms=transforms)
    train_loader, val_loader = train_val_split(dataset, seed=42, num_workers=0, batch_size=1)
    
    model = CustomUNet(4, input_dim=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print("Warming up GPU (10 batches)...")
    model.train()
    for i, (input, trg) in enumerate(train_loader):
        if i >= 10: break
        input, trg = input.to(device), trg.to(device)
        optimizer.zero_grad()
        loss = criterion(model(input), trg)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    TRAIN_BATCHES = 30
    VAL_BATCHES = 10
    
    print(f"Profiling {TRAIN_BATCHES} training batches...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=TRAIN_BATCHES, repeat=1),
        on_trace_ready=tensorboard_trace_handler('./bench_logs/train_profile')
    ) as prof:
        
        start_time = time.perf_counter()
        
        for i, (input, trg) in enumerate(train_loader):
            if i >= TRAIN_BATCHES: break
            
            input, trg = input.to(device), trg.to(device)
            optimizer.zero_grad()
            
            output = model(input)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            
            prof.step()
            
        torch.cuda.synchronize()
        end_time = time.perf_counter()

    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Time for {TRAIN_BATCHES} batches: {(end_time - start_time):.3f} s")
    print(f"  Avg Batch Time:      {((end_time - start_time) / 20) * 1000:.2f} ms")
    print(f"  Peak VRAM:           {peak_vram:.2f} MB\n")


    print("--- PART 3: Validation L1 Loss ---")
    model.eval()
    val_loss = 0.0
    
    print("Running validation passes...")
    with torch.no_grad():
        for i, (input, trg) in enumerate(val_loader):
            if i >= VAL_BATCHES:
                break
            
            input, trg = input.to(device), trg.to(device)
            output = model(input)
            loss = criterion(output, trg)
            val_loss += loss.item()
                
    avg_val_loss = val_loss / VAL_BATCHES
    print(f"  Validation L1 Loss (avg over {VAL_BATCHES} batches): {avg_val_loss:.5f}")
    print("\nNote: Run 'tensorboard --logdir optim/logs' to view the training trace.")

if __name__ == "__main__":
    main()