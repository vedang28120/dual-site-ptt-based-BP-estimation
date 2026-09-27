"""Anthropometrics and Pulse Wave Velocity (ePWV) Formulation.

This module computes the differential anatomical arterial path length between
the facial carotid ROI and the digital fingertip contact sensor:
    D_path = 0.55 * Height + 0.12 * ArmSpan (meters)

If ArmSpan is unmeasured, standard human biomechanical proportionality sets:
    ArmSpan = 1.03 * Height
    D_path = 0.55 * Height + 0.12 * (1.03 * Height) = 0.6736 * Height

Estimated Pulse Wave Velocity (ePWV):
    ePWV = D_path / Delta PTT (m/s)
"""

from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np


# Standard biomechanical constant ratios
ARM_SPAN_TO_HEIGHT_RATIO: float = 1.03
CAROTID_HEIGHT_FRACTION: float = 0.18
BRACHIAL_RADIAL_HEIGHT_FRACTION: float = 0.73
ARM_SPAN_FRACTION: float = 0.12


def compute_path_length(
    height_m: float,
    arm_span_m: Optional[float] = None,
) -> float:
    """Compute the anatomical differential travel path length D_path (meters).

    D_path represents the differential vascular distance:
        D_path = D_arm - D_carotid = 0.55 * Height + 0.12 * ArmSpan

    Parameters
    ----------
    height_m : float
        Subject height in meters. Must be in physiological range [0.5, 2.5].
    arm_span_m : Optional[float], optional
        Subject arm span (fingertip-to-fingertip) in meters.
        If None or non-positive, defaults to 1.03 * Height.

    Returns
    -------
    float
        Differential arterial path length D_path in meters.

    Raises
    ------
    ValueError
        If height_m <= 0 or outside plausible physiological boundaries.
    """
    if height_m <= 0.0:
        raise ValueError(f"Height must be strictly positive, got {height_m} m.")
    if height_m < 0.5 or height_m > 2.5:
        raise ValueError(f"Height {height_m} m is outside plausible human range [0.5, 2.5] m.")

    if arm_span_m is not None and arm_span_m > 0.0:
        if arm_span_m < 0.5 or arm_span_m > 3.0:
            raise ValueError(f"Arm span {arm_span_m} m is outside plausible range [0.5, 3.0] m.")
        effective_arm_span = float(arm_span_m)
    else:
        effective_arm_span = ARM_SPAN_TO_HEIGHT_RATIO * float(height_m)

    # 0.55 * Height + 0.12 * ArmSpan
    d_path = 0.55 * float(height_m) + ARM_SPAN_FRACTION * effective_arm_span
    return float(d_path)


def compute_epwv(
    d_path_m: float,
    delta_ptt_sec: float,
) -> float:
    """Compute estimated Pulse Wave Velocity (ePWV) from transit distance and time delay.

    ePWV = D_path / Delta PTT

    Parameters
    ----------
    d_path_m : float
        Arterial path length in meters.
    delta_ptt_sec : float
        Differential pulse transit time in seconds.

    Returns
    -------
    float
        Pulse wave velocity in meters per second (m/s).

    Raises
    ------
    ValueError
        If d_path_m <= 0 or delta_ptt_sec <= 0.
    """
    if d_path_m <= 0.0:
        raise ValueError(f"Arterial path length must be positive, got {d_path_m} m.")
    if delta_ptt_sec <= 0.0:
        raise ValueError(f"Pulse transit time must be strictly positive, got {delta_ptt_sec} s.")

    epwv = float(d_path_m) / float(delta_ptt_sec)
    return float(epwv)


def compute_epwv_from_profile(
    height_m: float,
    delta_ptt_sec: float,
    arm_span_m: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate path length and ePWV directly from anthropometric inputs and PTT.

    Parameters
    ----------
    height_m : float
        Subject height in meters.
    delta_ptt_sec : float
        Pulse transit time in seconds (or will convert if given in ms > 1.0).
    arm_span_m : Optional[float], optional
        Subject arm span in meters.

    Returns
    -------
    Dict[str, float]
        Dictionary with:
          - "d_path_m": Path length in meters.
          - "delta_ptt_sec": Pulse transit time in seconds.
          - "delta_ptt_ms": Pulse transit time in milliseconds.
          - "epwv_mps": Estimated pulse wave velocity in m/s.
    """
    # Auto-detect if user passed milliseconds (> 1.0)
    ptt_s = float(delta_ptt_sec)
    if ptt_s > 1.0:
        ptt_s /= 1000.0

    d_path = compute_path_length(height_m=height_m, arm_span_m=arm_span_m)
    epwv = compute_epwv(d_path_m=d_path, delta_ptt_sec=ptt_s)

    return {
        "d_path_m": d_path,
        "delta_ptt_sec": ptt_s,
        "delta_ptt_ms": ptt_s * 1000.0,
        "epwv_mps": epwv,
    }


class Anthropometrics:
    """Object-oriented helper for subject anthropometrics and arterial geometry."""

    def __init__(self, height_m: float, arm_span_m: Optional[float] = None) -> None:
        self.height_m = float(height_m)
        self.arm_span_m = float(arm_span_m) if arm_span_m is not None else None
        self.d_path_m = compute_path_length(self.height_m, self.arm_span_m)

    def calculate_epwv(self, delta_ptt_sec: float) -> float:
        return compute_epwv(self.d_path_m, delta_ptt_sec)

    def to_dict(self) -> Dict[str, Union[float, None]]:
        return {
            "height_m": self.height_m,
            "arm_span_m": self.arm_span_m,
            "d_path_m": self.d_path_m,
        }
