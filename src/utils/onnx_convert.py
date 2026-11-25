
import torch
import onnx
import numpy as np
from pathlib import Path
from collections import Counter

import tensorrt as trt
import modelopt.torch.opt as mto
from modelopt.torch.quantization.nn import TensorQuantizer

from src.model import CustomUNet

if __name__ == "__main__":
    checkpoint_path = Path("unet_int8_qat.pth")
    onnx_path = Path("model_int8.onnx")
    engine_path = Path("model_int8.engine")

    device = torch.device("cuda")

    model = CustomUNet(depth=4, input_dim=6)

    mto.restore(model, str(checkpoint_path))

    model = model.to(device).eval()

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            if "upsamplers" in name:
                module.disable()
                print("Disabled:", name)

    dummy_input = torch.randn(
        1, 6, 512, 512,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print("ONNX structure: OK")
    print(f"Saved: {onnx_path}")

    onnx_model = onnx.load(str(onnx_path))
    ops = Counter(node.op_type for node in onnx_model.graph.node)

    print("QuantizeLinear:", ops["QuantizeLinear"])
    print("DequantizeLinear:", ops["DequantizeLinear"])
    print("Q/DQ nodes:", ops["QuantizeLinear"] + ops["DequantizeLinear"])

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    print("\nParsing ONNX...")

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("TensorRT ONNX parsing failed")

    print("ONNX parsed successfully")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        1 << 30,
    )

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 6, 512, 512),
        opt=(1, 6, 512, 512),
        max=(4, 6, 512, 512),
    )

    config.add_optimization_profile(profile)

    print("\nBuilding TensorRT engine...")
    serialized_engine = builder.build_serialized_network(
        network,
        config,
    )

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.write_bytes(serialized_engine)
    print(f"TensorRT engine saved: {engine_path}")

    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError("Failed to load TensorRT engine")

    print("\nEngine loaded successfully")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)

        print(
            f"{name}: "
            f"shape={engine.get_tensor_shape(name)}, "
            f"dtype={engine.get_tensor_dtype(name)}, "
            f"mode={engine.get_tensor_mode(name)}"
        )

    print("\nDeployment artifacts created successfully!")