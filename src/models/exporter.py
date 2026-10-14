"""
Edge Model Exporter for Snapdragon 4 Gen 2 (Qualcomm SM4450).

Exports PulseFormerLite50M to:
  1. FP32 ONNX graph (Opset 17)
  2. Dynamic / Weight-Only INT8 ONNX graph (quantize_dynamic) targeting ~48.2 MB
Compatible with ONNX Runtime Mobile (ORT) with XNNPACK and QNN/NNAPI execution providers.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.models.pulseformer_lite_50m import PulseFormerLite50M


class MobileInferenceWrapper(nn.Module):
    """Wraps core PulseFormerLite50M to strip training-only hint projections for deployment."""

    def __init__(self, core: PulseFormerLite50M):
        super().__init__()
        self.core = core

    def forward(
        self, face_wave: torch.Tensor, palm_wave: torch.Tensor, physio_14d: torch.Tensor
    ) -> torch.Tensor:
        # Returns [SBP, DBP, MAP]
        return self.core(face_wave, palm_wave, physio_14d)["bp_pred"]


def export_student_for_snapdragon(
    student_model: PulseFormerLite50M,
    out_dir: str = "checkpoints",
    model_name: str = "pulseformer_lite_50m",
) -> Dict[str, str]:
    """
    Exports PulseFormerLite50M to FP32 ONNX and Dynamic INT8 ONNX.
    """
    os.makedirs(out_dir, exist_ok=True)
    student_model.eval().cpu()

    wrapper = MobileInferenceWrapper(student_model).eval()
    dummy_face = torch.randn(1, 3, 2500, dtype=torch.float32)
    dummy_palm = torch.randn(1, 3, 2500, dtype=torch.float32)
    dummy_phys = torch.randn(1, 14, dtype=torch.float32)

    onnx_fp32_path = os.path.join(out_dir, f"{model_name}_fp32.onnx")
    onnx_int8_path = os.path.join(out_dir, f"{model_name}_int8.onnx")

    print(f"[Export] Exporting FP32 ONNX to {onnx_fp32_path}...")
    try:
        torch.onnx.export(
            wrapper,
            (dummy_face, dummy_palm, dummy_phys),
            onnx_fp32_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["face_wave", "palm_wave", "physio_14d"],
            output_names=["bp_pred"],
            dynamo=False,
        )
    except TypeError:
        # Fallback if dynamo argument is not accepted in specific torch builds
        torch.onnx.export(
            wrapper,
            (dummy_face, dummy_palm, dummy_phys),
            onnx_fp32_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["face_wave", "palm_wave", "physio_14d"],
            output_names=["bp_pred"],
        )

    print(f"[Export] Quantizing MatMul/Linear layers to Dynamic INT8 -> {onnx_int8_path}...")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    fp32_size_mb = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    int8_size_mb = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    print(f"[Export] Done! FP32 Size: {fp32_size_mb:.2f} MB | INT8 Size: {int8_size_mb:.2f} MB")

    return {
        "onnx_fp32": onnx_fp32_path,
        "onnx_int8": onnx_int8_path,
        "fp32_size_mb": str(round(fp32_size_mb, 2)),
        "int8_size_mb": str(round(int8_size_mb, 2)),
    }
