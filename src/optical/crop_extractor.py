"""
64x64 Center Crop Direct Memory YUV_420_888 Extraction for Poco M6 Pro 5G.

Directly samples the central digital arterial vascular bed under LED torch transillumination:
- Bounding Box: X in [288..351], Y in [208..271] on 640x480 frame (4,096 pixels, 1.33% of frame).
- Avoids full-frame ISP memory bandwidth and bypasses lateral specular reflection and tissue edge leakage.
- Zero-allocation spatial averaging for Green and Red channels.
- Enforces linear transillumination regime: 120 <= mean(Green) <= 190, avoiding CMOS saturation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import numpy as np


@dataclass(frozen=True)
class CropMetrics:
    """Quantitative metrics extracted from the 64x64 center crop."""
    mean_r: float
    mean_g: float
    mean_b: float
    saturated_pixel_count: int
    saturated_fraction: float
    is_saturated: bool
    is_underexposed: bool
    in_optimal_regime: bool
    pixel_count: int = 4096

    @property
    def rgb_vector(self) -> np.ndarray:
        return np.array([self.mean_r, self.mean_g, self.mean_b], dtype=np.float32)


def direct_yuv420_to_rgb(
    y_val: int,
    u_val: int,
    v_val: int,
) -> Tuple[float, float, float]:
    """
    ITU-R BT.601 scalar conversion from YUV (with centered U/V chrominance) to RGB.
    
    Y in [0, 255], U in [-128, 127], V in [-128, 127].
    """
    r = max(0.0, min(255.0, y_val + 1.402 * v_val))
    g = max(0.0, min(255.0, y_val - 0.344136 * u_val - 0.714136 * v_val))
    b = max(0.0, min(255.0, y_val + 1.772 * u_val))
    return r, g, b


def compute_perfusion_index(ac_amplitude: float, dc_baseline: float) -> float:
    """
    Compute optical Perfusion Index (PI): PI = (AC / DC) * 100%.
    Typical digital artery contact PPG PI ranges from 0.5% to 5.0%.
    """
    if dc_baseline <= 1e-6:
        return 0.0
    return float((ac_amplitude / dc_baseline) * 100.0)


class CenterCropExtractor:
    """
    High-efficiency 64x64 center crop extractor for 480p (640x480) contact PPG frames.
    """
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        crop_size: int = 64,
        saturation_ceiling: float = 250.0,
        optimal_min_g: float = 120.0,
        optimal_max_g: float = 190.0,
        underexposure_floor: float = 80.0,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.crop_size = crop_size
        self.saturation_ceiling = saturation_ceiling
        self.optimal_min_g = optimal_min_g
        self.optimal_max_g = optimal_max_g
        self.underexposure_floor = underexposure_floor

        # Compute exact symmetric center crop coordinates
        self.crop_x_start = (frame_width - crop_size) // 2      # 288 for 640
        self.crop_x_end = self.crop_x_start + crop_size         # 352 for 640
        self.crop_y_start = (frame_height - crop_size) // 2     # 208 for 480
        self.crop_y_end = self.crop_y_start + crop_size         # 272 for 480
        self.num_pixels = crop_size * crop_size                 # 4,096

    def get_crop_bounds(self) -> Tuple[int, int, int, int]:
        """Return (x_start, x_end, y_start, y_end)."""
        return self.crop_x_start, self.crop_x_end, self.crop_y_start, self.crop_y_end

    def extract_from_rgb(self, rgb_image: np.ndarray) -> CropMetrics:
        """
        Extract spatial average and quality metrics from an RGB numpy array (H, W, 3).
        """
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected image of shape (H, W, 3), got {rgb_image.shape}")

        h, w, _ = rgb_image.shape
        if h != self.frame_height or w != self.frame_width:
            raise ValueError(f"Frame dimension mismatch: expected ({self.frame_height}, {self.frame_width}), got ({h}, {w})")

        crop = rgb_image[
            self.crop_y_start:self.crop_y_end,
            self.crop_x_start:self.crop_x_end,
            :
        ].astype(np.float32)

        mean_r = float(np.mean(crop[:, :, 0]))
        mean_g = float(np.mean(crop[:, :, 1]))
        mean_b = float(np.mean(crop[:, :, 2]))

        # Count saturated pixels (photodiode ceiling blowout)
        saturated_mask = (crop[:, :, 1] >= self.saturation_ceiling) | (crop[:, :, 0] >= self.saturation_ceiling)
        sat_count = int(np.sum(saturated_mask))
        sat_fraction = float(sat_count / self.num_pixels)

        is_saturated = (mean_g > 220.0) or (sat_fraction > 0.02)
        is_underexposed = mean_g < self.underexposure_floor
        in_optimal_regime = (self.optimal_min_g <= mean_g <= self.optimal_max_g) and (sat_fraction == 0.0)

        return CropMetrics(
            mean_r=mean_r,
            mean_g=mean_g,
            mean_b=mean_b,
            saturated_pixel_count=sat_count,
            saturated_fraction=sat_fraction,
            is_saturated=is_saturated,
            is_underexposed=is_underexposed,
            in_optimal_regime=in_optimal_regime,
            pixel_count=self.num_pixels,
        )

    def extract_crop_patch(self, rgb_image: np.ndarray) -> np.ndarray:
        """Extract the raw (64, 64, 3) cropped image array."""
        return rgb_image[
            self.crop_y_start:self.crop_y_end,
            self.crop_x_start:self.crop_x_end,
            :
        ].copy()

    def extract_from_yuv_planes(
        self,
        y_plane: bytes,
        u_plane: bytes,
        v_plane: bytes,
        y_row_stride: int,
        uv_row_stride: int,
        uv_pixel_stride: int = 1,
    ) -> CropMetrics:
        """
        Simulate direct memory plane extraction matching Android Camera2 ImageReader YUV_420_888.
        Zero allocation inside inner loop, scalar accumulation.
        """
        sum_r = 0.0
        sum_g = 0.0
        sum_b = 0.0
        sat_count = 0

        for y in range(self.crop_y_start, self.crop_y_end):
            y_offset = y * y_row_stride
            uv_offset = (y >> 1) * uv_row_stride

            for x in range(self.crop_x_start, self.crop_x_end):
                y_idx = y_offset + x
                uv_idx = uv_offset + (x >> 1) * uv_pixel_stride

                y_val = y_plane[y_idx]
                u_val = u_plane[uv_idx] - 128
                v_val = v_plane[uv_idx] - 128

                r = max(0.0, min(255.0, y_val + 1.402 * v_val))
                g = max(0.0, min(255.0, y_val - 0.344136 * u_val - 0.714136 * v_val))
                b = max(0.0, min(255.0, y_val + 1.772 * u_val))

                if g >= self.saturation_ceiling or r >= self.saturation_ceiling:
                    sat_count += 1

                sum_r += r
                sum_g += g
                sum_b += b

        mean_r = sum_r / self.num_pixels
        mean_g = sum_g / self.num_pixels
        mean_b = sum_b / self.num_pixels

        sat_fraction = sat_count / self.num_pixels
        is_saturated = (mean_g > 220.0) or (sat_fraction > 0.02)
        is_underexposed = mean_g < self.underexposure_floor
        in_optimal_regime = (self.optimal_min_g <= mean_g <= self.optimal_max_g) and (sat_fraction == 0.0)

        return CropMetrics(
            mean_r=float(mean_r),
            mean_g=float(mean_g),
            mean_b=float(mean_b),
            saturated_pixel_count=sat_count,
            saturated_fraction=float(sat_fraction),
            is_saturated=is_saturated,
            is_underexposed=is_underexposed,
            in_optimal_regime=in_optimal_regime,
            pixel_count=self.num_pixels,
        )


def extract_64x64_center_crop(rgb_frame: np.ndarray) -> np.ndarray:
    """Convenience functional interface returning 3-element float32 array [R, G, B]."""
    extractor = CenterCropExtractor()
    metrics = extractor.extract_from_rgb(rgb_frame)
    return metrics.rgb_vector
