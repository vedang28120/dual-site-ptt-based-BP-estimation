"""
src/mobile: Mobile System Integration & Feasibility Pipeline for Poco M6 Pro 5G.

Provides:
- TFLiteRunner: Static-buffered mobile INT8 inference runner for Snapdragon 4 Gen 2 (SM4450).
- DualSiteBPPipeline: Complete end-to-end BP pipeline bridging Camera2/PCHIP optical synchronizer,
                      HRVP-PINN hemodynamic solver, and DS-TCN deep learning model.
- PipelineConfig, PipelineResult, SignalQualityMetrics, EdgeRuntimeProfile.
"""

from src.mobile.tflite_runner import (
    TFLiteRunner,
    EdgeRuntimeProfile,
)
from src.mobile.pipeline import (
    DualSiteBPPipeline,
    PipelineConfig,
    PipelineResult,
    SignalQualityMetrics,
    estimate_snr_db,
    normalize_waveform_tensor,
)

__all__ = [
    "TFLiteRunner",
    "EdgeRuntimeProfile",
    "DualSiteBPPipeline",
    "PipelineConfig",
    "PipelineResult",
    "SignalQualityMetrics",
    "estimate_snr_db",
    "normalize_waveform_tensor",
]
