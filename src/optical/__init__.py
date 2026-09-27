"""
Dual-Site Optical Engine Module for Poco M6 Pro 5G / Android Camera2.

Provides:
- Camera hardware profiles and manual exposure anti-saturation configuration.
- 64x64 center crop direct YUV extraction with zero-allocation spatial averaging.
- Decoupled fast-path face ROI tracking with EMA smoothing and EAR blink gating.
- Nanosecond hardware monotonic clock binding and PCHIP master time-grid resampling.
"""

from src.optical.camera_config import (
    CameraStreamConfig,
    ManualExposureConfig,
    DynamicExposureController,
    DualCameraHardwareProfile,
    validate_hardware_profile,
    create_rear_camera_capture_params,
    create_front_camera_capture_params,
)

from src.optical.crop_extractor import (
    CenterCropExtractor,
    CropMetrics,
    direct_yuv420_to_rgb,
    extract_64x64_center_crop,
    compute_perfusion_index,
)

from src.optical.face_roi_tracker import (
    FaceRoiTracker,
    FaceRoiBoundingBoxes,
    HeadPoseAngles,
    compute_eye_aspect_ratio,
    estimate_head_pose_angles,
)

from src.optical.time_grid_synchronizer import (
    TimeGridSynchronizer,
    SynchronizationResult,
    compute_timestamp_jitter,
)

__all__ = [
    "CameraStreamConfig",
    "ManualExposureConfig",
    "DynamicExposureController",
    "DualCameraHardwareProfile",
    "validate_hardware_profile",
    "create_rear_camera_capture_params",
    "create_front_camera_capture_params",
    "CenterCropExtractor",
    "CropMetrics",
    "direct_yuv420_to_rgb",
    "extract_64x64_center_crop",
    "compute_perfusion_index",
    "FaceRoiTracker",
    "FaceRoiBoundingBoxes",
    "HeadPoseAngles",
    "compute_eye_aspect_ratio",
    "estimate_head_pose_angles",
    "TimeGridSynchronizer",
    "SynchronizationResult",
    "compute_timestamp_jitter",
]
