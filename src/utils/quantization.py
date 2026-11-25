import torch
import torch.nn as nn
from torchvision.transforms import Resize

import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

from src.model import CustomUNet, load_checkpoint
from src.preprocessing import (
    LidarImgDataset,
    train_val_split,
    TRAIN_DIR,
    CACHE_DIR,
)

CHECKPOINT = "best_model.pt"

PTQ_SAVE_PATH = "unet_int8_ptq.pth"
QAT_SAVE_PATH = "unet_int8_qat.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CALIBRATION_SAMPLES = 256
QAT_ENABLED = True
QAT_EPOCHS = 5
QAT_LR = 1e-5

def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)

    arch = checkpoint.get("arch", {"depth": 4, "input_dim": 6})

    model = CustomUNet(**arch).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    return model

def get_batch(batch):
    if isinstance(batch, (tuple, list)):
        x, target = batch[0], batch[1]
    elif isinstance(batch, dict):
        x = batch["input"]
        target = batch["target"]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    return x.to(DEVICE), target.to(DEVICE)


@torch.inference_mode()
def evaluate(model, loader, name):
    model.eval()

    criterion = nn.L1Loss(reduction="sum")

    total_error = 0.0
    total_elements = 0

    for batch in loader:
        x, target = get_batch(batch)

        prediction = model(x)

        if prediction.shape != target.shape:
            raise RuntimeError(
                f"{name}: output shape {prediction.shape} "
                f"does not match target shape {target.shape}"
            )

        total_error += criterion(prediction, target).item()
        total_elements += target.numel()

    mean_l1 = total_error / total_elements

    print(f"{name} mean L1: {mean_l1:.8f}")
    return mean_l1


def calibrate(model, loader):
    model.eval()

    samples_seen = 0

    with torch.inference_mode():
        for batch in loader:
            x, _ = get_batch(batch)

            model(x)

            samples_seen += x.shape[0]

            if samples_seen >= CALIBRATION_SAMPLES:
                break

    print(f"Calibration samples: {samples_seen}")


def train_qat(model, train_loader, val_loader):
    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=QAT_LR,
    )

    for epoch in range(QAT_EPOCHS):
        model.train()

        running_loss = 0.0
        batches = 0

        for batch in train_loader:
            x, target = get_batch(batch)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(x)
            loss = criterion(prediction, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        train_loss = running_loss / max(batches, 1)

        print(
            f"QAT epoch {epoch + 1}/{QAT_EPOCHS} "
            f"- train L1: {train_loss:.8f}"
        )

        evaluate(model, val_loader, f"QAT epoch {epoch + 1}")

    return model


def main():
    print(f"Device: {DEVICE}")

    dataset = LidarImgDataset(
        sample_path=TRAIN_DIR,
        cache_path=CACHE_DIR,
        transforms=nn.Sequential(Resize((512, 512))),
    )

    train_loader, val_loader = train_val_split(
        dataset,
        seed=42,
        num_workers=0,
        batch_size=1,
    )
    
    print("\n===== FP32 BASELINE =====")

    fp32_model = load_model()
    fp32_l1 = evaluate(fp32_model, val_loader, "FP32")

    print("\n===== INT8 PTQ =====")

    ptq_model = load_model()
    ptq_model.eval()

    def calibration_loop(model):
        calibrate(model, train_loader)

    ptq_model = mtq.quantize(
        ptq_model,
        mtq.INT8_DEFAULT_CFG,
        forward_loop=calibration_loop,
    )

    mtq.print_quant_summary(ptq_model)

    ptq_l1 = evaluate(ptq_model, val_loader, "INT8 PTQ")

    print(
        f"PTQ L1 change: {ptq_l1 - fp32_l1:+.8f} "
        f"({(ptq_l1 / fp32_l1 - 1) * 100:+.2f}%)"
    )

    mto.save(ptq_model, PTQ_SAVE_PATH)
    print(f"Saved PTQ model: {PTQ_SAVE_PATH}")

    print("\n===== INT8 QAT =====")

    qat_model = ptq_model

    qat_model = train_qat(
        qat_model,
        train_loader,
        val_loader,
    )

    qat_l1 = evaluate(qat_model, val_loader, "INT8 QAT")

    print(
        f"QAT L1 change vs FP32: {qat_l1 - fp32_l1:+.8f} "
        f"({(qat_l1 / fp32_l1 - 1) * 100:+.2f}%)"
    )

    mto.save(qat_model, QAT_SAVE_PATH)
    print(f"Saved QAT model: {QAT_SAVE_PATH}")

    print("\n===== SUMMARY =====")
    print(f"FP32:    {fp32_l1:.8f}")
    print(f"INT8 PTQ: {ptq_l1:.8f}")
    print(f"INT8 QAT: {qat_l1:.8f}")


if __name__ == "__main__":
    main()
