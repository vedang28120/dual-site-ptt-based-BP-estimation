"""
Camera Configuration and Manual Exposure Anti-Saturation Protocol for Poco M6 Pro 5G.

Designed for Qualcomm Snapdragon 4 Gen 2 (SM4450) with Qualcomm Spectra™ Dual 12-bit ISP.
Configures Android Camera2 dual-stream capture (480p @ 30 FPS across front/rear):
- Rear Contact PPG: Manual exposure (ISO 50 floor, 1.0 ms shutter), locked focus, torch mode.
- Front Facial rPPG: Auto-exposure (fixed 30 FPS target range), fast color correction.
- Closed-loop dynamic anti-saturation safety guard for transillumination stabilization.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List


class CameraId(str, Enum):
    REAR = "0"
    FRONT = "1"


class TimestampSource(str, Enum):
    REALTIME = "REALTIME"  # Linux CLOCK_BOOTTIME (SystemClock.elapsedRealtimeNanos)
    UNKNOWN = "UNKNOWN"


class AEControlMode(str, Enum):
    OFF = "CONTROL_AE_MODE_OFF"
    ON = "CONTROL_AE_MODE_ON"
    AUTO = "CONTROL_AE_MODE_AUTO"


class AFControlMode(str, Enum):
    OFF = "CONTROL_AF_MODE_OFF"
    AUTO = "CONTROL_AF_MODE_AUTO"
    CONTINUOUS_PICTURE = "CONTROL_AF_MODE_CONTINUOUS_PICTURE"


class FlashMode(str, Enum):
    OFF = "FLASH_MODE_OFF"
    SINGLE = "FLASH_MODE_SINGLE"
    TORCH = "FLASH_MODE_TORCH"


@dataclass(frozen=True)
class CameraStreamConfig:
    """Stream resolution and queue configuration for Camera2 ImageReader."""
    camera_id: CameraId
    width: int = 640
    height: int = 480
    format: str = "YUV_420_888"
    fps_range: Tuple[int, int] = (30, 30)
    max_images: int = 2  # Limits buffer lag and GC pressure on Android HAL

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    @property
    def pixel_rate_mp_per_sec(self) -> float:
        return (self.total_pixels * self.fps_range[1]) / 1e6


@dataclass
class ManualExposureConfig:
    """Manual exposure and sensor parameter configuration for rear contact PPG."""
    ae_mode: AEControlMode = AEControlMode.OFF
    iso: int = 50  # ISO 50 floor: suppresses thermal shot noise and maximizes full-well dynamic range
    exposure_time_ns: int = 1_000_000  # 1.0 ms (1/1000s) default transillumination shutter speed
    min_exposure_ns: int = 200_000     # 0.2 ms safety floor
    max_exposure_ns: int = 10_000_000  # 10.0 ms upper limit
    iso_range: Tuple[int, int] = (50, 3200)
    af_mode: AFControlMode = AFControlMode.OFF
    lens_focus_distance: float = 0.0  # Fixed infinity (0.0 diopters) disables VCM hunting
    flash_mode: FlashMode = FlashMode.TORCH
    awb_lock: bool = True
    noise_reduction_mode: str = "OFF"
    edge_mode: str = "OFF"
    color_correction_mode: str = "FAST"

    def clamp_exposure(self, requested_ns: int) -> int:
        """Clamp requested exposure time within hardware safety limits."""
        return max(self.min_exposure_ns, min(self.max_exposure_ns, requested_ns))


@dataclass
class FrontCameraConfig:
    """Front camera configuration for facial rPPG."""
    ae_mode: AEControlMode = AEControlMode.ON
    af_mode: AFControlMode = AFControlMode.CONTINUOUS_PICTURE
    awb_mode: str = "AUTO"
    noise_reduction_mode: str = "OFF"
    edge_mode: str = "OFF"
    color_correction_mode: str = "FAST"
    fps_range: Tuple[int, int] = (30, 30)


@dataclass
class DualCameraHardwareProfile:
    """
    Hardware and HAL capability profile for the target platform (Xiaomi Poco M6 Pro 5G).
    Snapdragon 4 Gen 2 (SM4450) with Qualcomm Spectra Dual 12-bit ISP.
    """
    device_model: str = "Xiaomi Poco M6 Pro 5G"
    chipset: str = "Qualcomm Snapdragon 4 Gen 2 (SM4450)"
    isp_name: str = "Qualcomm Spectra Dual 12-bit ISP"
    isp_max_throughput_mp_per_sec: float = 32.0  # Up to 2x 16MP @ 30 FPS concurrent
    rear_sensor: str = "Samsung ISOCELL S5KJN1 (50 MP, 1/2.76\")"
    front_sensor: str = "OmniVision OV08D10 (8 MP, 1/4.0\")"
    concurrent_streaming_supported: bool = True
    timestamp_source: TimestampSource = TimestampSource.REALTIME
    concurrent_camera_ids: List[List[str]] = field(
        default_factory=lambda: [["0", "1"]]
    )

    def compute_bandwidth_headroom(self, front_stream: CameraStreamConfig, rear_stream: CameraStreamConfig) -> float:
        """Compute percentage ISP throughput headroom under concurrent streaming."""
        required_mp_per_sec = front_stream.pixel_rate_mp_per_sec + rear_stream.pixel_rate_mp_per_sec
        headroom_percent = ((self.isp_max_throughput_mp_per_sec - required_mp_per_sec) / self.isp_max_throughput_mp_per_sec) * 100.0
        return headroom_percent


class DynamicExposureController:
    """
    Closed-loop dynamic exposure safety guard for contact PPG.
    
    Under flash LED transillumination, photodiode full-well capacity can saturate
    at 255 (8-bit ADC ceiling), causing AC pulsatile amplitude to collapse to zero.
    This controller monitors mean green channel intensity:
    - If mean_green > 220.0 (saturation threat), decrements exposure time by 20%.
    - If mean_green < 80.0 (underexposure), increments exposure time by 20%.
    - Target linear transillumination regime: 120.0 <= mean_green <= 190.0.
    - Permanently locks exposure after convergence or reaching calibration budget.
    """
    def __init__(
        self,
        config: Optional[ManualExposureConfig] = None,
        target_min: float = 120.0,
        target_max: float = 190.0,
        saturation_threshold: float = 220.0,
        underexposure_threshold: float = 80.0,
        adjustment_factor: float = 0.20,
        max_calibration_frames: int = 10,
    ):
        self.config = config or ManualExposureConfig()
        self.target_min = target_min
        self.target_max = target_max
        self.saturation_threshold = saturation_threshold
        self.underexposure_threshold = underexposure_threshold
        self.adjustment_factor = adjustment_factor
        self.max_calibration_frames = max_calibration_frames

        self.current_exposure_ns = self.config.exposure_time_ns
        self.frame_count = 0
        self.is_locked = False
        self.history: List[Dict[str, Any]] = []

    def update(self, mean_green: float) -> Tuple[int, bool]:
        """
        Process incoming frame's mean green intensity.
        
        Returns:
            Tuple[int, bool]: (updated_exposure_time_ns, is_permanently_locked)
        """
        if self.is_locked:
            return self.current_exposure_ns, True

        self.frame_count += 1
        previous_exposure = self.current_exposure_ns
        action = "STABLE"

        if mean_green > self.saturation_threshold:
            # Saturation threat: decrease shutter time to bring intensity down
            self.current_exposure_ns = int(self.current_exposure_ns * (1.0 - self.adjustment_factor))
            action = "DECREASE_EXPOSURE"
        elif mean_green < self.underexposure_threshold:
            # Underexposed: increase shutter time to enter transillumination zone
            self.current_exposure_ns = int(self.current_exposure_ns * (1.0 + self.adjustment_factor))
            action = "INCREASE_EXPOSURE"
        elif self.target_min <= mean_green <= self.target_max:
            # In optimal linear regime: lock immediately
            self.is_locked = True
            action = "LOCK_OPTIMAL"

        # Hardware safety clamping
        self.current_exposure_ns = self.config.clamp_exposure(self.current_exposure_ns)

        if self.frame_count >= self.max_calibration_frames:
            self.is_locked = True
            if action != "LOCK_OPTIMAL":
                action = "LOCK_MAX_FRAMES"

        self.history.append({
            "frame": self.frame_count,
            "mean_green": mean_green,
            "previous_exposure_ns": previous_exposure,
            "new_exposure_ns": self.current_exposure_ns,
            "action": action,
            "locked": self.is_locked,
        })

        return self.current_exposure_ns, self.is_locked

    def reset(self) -> None:
        """Reset the controller for a new acquisition session."""
        self.current_exposure_ns = self.config.exposure_time_ns
        self.frame_count = 0
        self.is_locked = False
        self.history.clear()


def validate_hardware_profile(profile: DualCameraHardwareProfile) -> Dict[str, Any]:
    """
    Validate that hardware profile meets all criteria for concurrent dual-site acquisition.
    """
    validations = {
        "concurrent_supported": profile.concurrent_streaming_supported,
        "timestamp_realtime": profile.timestamp_source == TimestampSource.REALTIME,
        "contains_rear_and_front": any(
            CameraId.REAR.value in s and CameraId.FRONT.value in s for s in profile.concurrent_camera_ids
        ),
        "isp_headroom_sufficient": profile.isp_max_throughput_mp_per_sec >= 16.0,
    }
    all_passed = all(validations.values())
    return {
        "valid": all_passed,
        "checks": validations,
        "device": profile.device_model,
        "chipset": profile.chipset,
    }


def create_rear_camera_capture_params(config: ManualExposureConfig) -> Dict[str, Any]:
    """
    Construct Android Camera2 CaptureRequest parameter dictionary for rear contact PPG.
    """
    return {
        "CONTROL_MODE": "CONTROL_MODE_AUTO",
        "CONTROL_AE_MODE": config.ae_mode.value,
        "SENSOR_SENSITIVITY": config.iso,
        "SENSOR_EXPOSURE_TIME": config.exposure_time_ns,
        "CONTROL_AF_MODE": config.af_mode.value,
        "LENS_FOCUS_DISTANCE": config.lens_focus_distance,
        "FLASH_MODE": config.flash_mode.value,
        "CONTROL_AWB_MODE": "CONTROL_AWB_MODE_AUTO",
        "CONTROL_AWB_LOCK": config.awb_lock,
        "NOISE_REDUCTION_MODE": f"NOISE_REDUCTION_MODE_{config.noise_reduction_mode}",
        "EDGE_MODE": f"EDGE_MODE_{config.edge_mode}",
        "COLOR_CORRECTION_MODE": f"COLOR_CORRECTION_MODE_{config.color_correction_mode}",
        "CONTROL_AE_TARGET_FPS_RANGE": [30, 30],
    }


def create_front_camera_capture_params(config: FrontCameraConfig) -> Dict[str, Any]:
    """
    Construct Android Camera2 CaptureRequest parameter dictionary for front facial rPPG.
    """
    return {
        "CONTROL_MODE": "CONTROL_MODE_AUTO",
        "CONTROL_AE_MODE": config.ae_mode.value,
        "CONTROL_AWB_MODE": f"CONTROL_AWB_MODE_{config.awb_mode}",
        "CONTROL_AF_MODE": config.af_mode.value,
        "NOISE_REDUCTION_MODE": f"NOISE_REDUCTION_MODE_{config.noise_reduction_mode}",
        "EDGE_MODE": f"EDGE_MODE_{config.edge_mode}",
        "COLOR_CORRECTION_MODE": f"COLOR_CORRECTION_MODE_{config.color_correction_mode}",
        "CONTROL_AE_TARGET_FPS_RANGE": list(config.fps_range),
    }
