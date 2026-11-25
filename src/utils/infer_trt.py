import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cuda

import torch
import torch.nn as nn
from pathlib import Path

from src.projections import build_lidar_projections


class TensorRTPredictor:
    def __init__(self, engine_path: Path):
        self._check(cuda.cuInit(0), "cuInit")
        _, device = self._check_result(cuda.cuDeviceGet(0), "cuDeviceGet")
        _, current_ctx = self._check_result(cuda.cuCtxGetCurrent(), "cuCtxGetCurrent")

        if current_ctx is None:
            _, self.cuda_context = self._check_result(cuda.cuDevicePrimaryCtxRetain(device), "cuDevicePrimaryCtxRetain")
            self._check(cuda.cuCtxSetCurrent(self.cuda_context), "cuCtxSetCurrent")
        else:
            self.cuda_context = current_ctx

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        _, self.stream = self._check_result(cuda.cuStreamCreate(0), "cuStreamCreate")

        self.input_name = next(
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(
                self.engine.get_tensor_name(i)
            ) == trt.TensorIOMode.INPUT
        )

        self.output_name = next(
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(
                self.engine.get_tensor_name(i)
            ) == trt.TensorIOMode.OUTPUT
        )

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        print(f"TensorRT version: {trt.__version__}")
        print(f"Input: {self.input_name}, dtype={self.input_dtype}")
        print(f"Output: {self.output_name}, dtype={self.output_dtype}")

    @staticmethod
    def _check(result, operation: str):
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA operation failed: {operation}")

    @classmethod
    def _check_result(cls, result, operation: str):
        status, *values = result
        cls._check(status, operation)
        return (status, *values)

    def predict(self, sample_path: Path, device: torch.device, transforms: nn.Sequential) -> torch.Tensor:
        sample_input = build_lidar_projections(sample_path, device, transforms)

        if sample_input is None:
            raise RuntimeError(f"Preprocessing produced no tensor for {sample_path}")

        input_np = sample_input.detach().cpu().numpy()[None, ...]
        input_np = np.ascontiguousarray(input_np.astype(self.input_dtype))

        if not self.context.set_input_shape(self.input_name, input_np.shape):
            raise RuntimeError(f"TensorRT rejected input shape: {input_np.shape}")

        output_shape = tuple(self.context.get_tensor_shape(self.output_name))

        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(f"Output shape is not resolved: {output_shape}")

        output_np = np.empty(output_shape, dtype=self.output_dtype)

        _, d_input = self._check_result(cuda.cuMemAlloc(input_np.nbytes), "cuMemAlloc input")

        _, d_output = self._check_result(cuda.cuMemAlloc(output_np.nbytes), "cuMemAlloc output")

        try:
            self.context.set_tensor_address(self.input_name, int(d_input))
            self.context.set_tensor_address(self.output_name, int(d_output))

            self._check(
                cuda.cuMemcpyHtoDAsync(
                    d_input,
                    input_np,
                    input_np.nbytes,
                    self.stream,
                ),
                "cuMemcpyHtoDAsync",
            )

            if not self.context.execute_async_v3(int(self.stream)):
                raise RuntimeError("TensorRT inference failed")

            self._check(
                cuda.cuMemcpyDtoHAsync(
                    output_np,
                    d_output,
                    output_np.nbytes,
                    self.stream,
                ), "cuMemcpyDtoHAsync"
            )

            self._check(cuda.cuStreamSynchronize(self.stream), "cuStreamSynchronize")

        finally:
            self._check(cuda.cuMemFree(d_input), "cuMemFree input")
            self._check(cuda.cuMemFree(d_output), "cuMemFree output")

        return torch.from_numpy(output_np).squeeze(0)