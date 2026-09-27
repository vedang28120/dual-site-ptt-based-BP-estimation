"""IMU-Guided Two-Elevation Hydrostatic Perturbation Engine.

This module models the hydrostatic fluid column pressure perturbation:
    Delta P_hydro = rho_blood * g * Delta h

Where:
    rho_blood = 1055 kg/m^3 (density of whole human blood at 37 C)
    g = 9.80665 m/s^2 (standard gravitational acceleration)
    1 mmHg = 133.322387415 Pa
    => Delta P_hydro [mmHg] = 0.077601 * Delta h [mm] = 77.6014 * Delta h [m]

It also provides IMU kinematic double integration with Zero-Velocity Updates (ZUPT)
to reliably track smartphone elevation shifts between Elevation 1 and Elevation 2.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np


# Standard physical constants
RHO_BLOOD: float = 1055.0  # kg/m^3
G_GRAVITY: float = 9.80665  # m/s^2
PA_TO_MMHG: float = 1.0 / 133.322387415  # 0.00750061683 mmHg/Pa
MMHG_PER_METER: float = RHO_BLOOD * G_GRAVITY * PA_TO_MMHG  # 77.60142 mmHg/m
MMHG_PER_MM: float = MMHG_PER_METER / 1000.0  # 0.07760142 mmHg/mm


def elevation_to_hydrostatic_pressure(
    delta_h_m: float,
) -> Dict[str, float]:
    """Convert vertical elevation displacement to hydrostatic column pressure.

    Delta P_hydro = rho_blood * g * Delta h

    Parameters
    ----------
    delta_h_m : float
        Vertical displacement in meters.
        Positive indicates hand raised relative to heart level (decreasing digital pressure).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
          - "delta_h_m": Displacement in meters.
          - "delta_h_mm": Displacement in millimeters.
          - "delta_p_pa": Hydrostatic pressure shift in Pascals (Pa).
          - "delta_p_mmhg": Hydrostatic pressure shift in mmHg.
    """
    h_m = float(delta_h_m)
    p_pa = RHO_BLOOD * G_GRAVITY * h_m
    p_mmhg = p_pa * PA_TO_MMHG

    return {
        "delta_h_m": h_m,
        "delta_h_mm": h_m * 1000.0,
        "delta_p_pa": float(p_pa),
        "delta_p_mmhg": float(p_mmhg),
    }


def pressure_to_elevation(
    delta_p_mmhg: float,
) -> float:
    """Convert hydrostatic pressure shift in mmHg back to vertical elevation displacement (meters).

    Parameters
    ----------
    delta_p_mmhg : float
        Pressure shift in mmHg.

    Returns
    -------
    float
        Vertical displacement Delta h in meters.
    """
    p_pa = float(delta_p_mmhg) / PA_TO_MMHG
    delta_h = p_pa / (RHO_BLOOD * G_GRAVITY)
    return float(delta_h)


def integrate_zupt_elevation(
    time_sec: np.ndarray,
    accel_vertical: np.ndarray,
    stationary_threshold: float = 0.06,
    min_hold_duration_sec: float = 1.0,
) -> Dict[str, Union[float, np.ndarray, bool]]:
    """Estimate vertical elevation displacement Delta h via double integration with ZUPT.

    Zero-Velocity Updates (ZUPT) detect stationary periods where vertical acceleration
    magnitude is below threshold, clamping linear velocity drift to zero.

    Parameters
    ----------
    time_sec : np.ndarray
        Array of timestamps in seconds.
    accel_vertical : np.ndarray
        Linear vertical acceleration in m/s^2 (gravity subtracted).
    stationary_threshold : float, optional
        Acceleration threshold for stationary detection in m/s^2 (default: 0.06).
    min_hold_duration_sec : float, optional
        Minimum duration in seconds to consider a hold stable (default: 1.0 s).

    Returns
    -------
    Dict[str, Union[float, np.ndarray, bool]]
        Dictionary containing:
          - "delta_h_m": Total elevation displacement between initial and final hold (m).
          - "delta_p_mmhg": Corresponding hydrostatic pressure shift (mmHg).
          - "velocity_m_s": Integrated velocity time series (m/s).
          - "elevation_m": Integrated vertical position time series (m).
          - "hold_stable": Boolean indicating whether stationary holds were identified.
    """
    t = np.asarray(time_sec, dtype=np.float64)
    a = np.asarray(accel_vertical, dtype=np.float64)

    if len(t) < 5:
        return {
            "delta_h_m": 0.0,
            "delta_p_mmhg": 0.0,
            "velocity_m_s": np.zeros_like(t),
            "elevation_m": np.zeros_like(t),
            "hold_stable": False,
        }

    dt = np.diff(t)
    dt = np.insert(dt, 0, dt[0] if len(dt) > 0 else 0.01)

    # Detect stationary samples: |a| < threshold
    is_stationary = np.abs(a) < stationary_threshold

    # Integrate acceleration to velocity with trapezoidal integration
    n = len(t)
    v = np.zeros(n, dtype=np.float64)
    h = np.zeros(n, dtype=np.float64)

    # Initial pass: cumulative trapezoidal integration
    current_v = 0.0
    for i in range(1, n):
        step_dt = t[i] - t[i - 1]
        if is_stationary[i] and is_stationary[i - 1]:
            # ZUPT: Zero velocity update
            current_v = 0.0
        else:
            current_v += 0.5 * (a[i] + a[i - 1]) * step_dt
        v[i] = current_v

    # Second pass: integrate velocity to elevation
    for i in range(1, n):
        step_dt = t[i] - t[i - 1]
        h[i] = h[i - 1] + 0.5 * (v[i] + v[i - 1]) * step_dt

    # Identify initial hold (first 1.0s or stationary block) and final hold
    # Initial hold segment
    init_mask = is_stationary[: max(1, int(n * 0.25))]
    final_mask = is_stationary[int(n * 0.75) :]

    init_hold_stable = np.mean(init_mask) > 0.5 if len(init_mask) > 0 else False
    final_hold_stable = np.mean(final_mask) > 0.5 if len(final_mask) > 0 else False
    hold_stable = bool(init_hold_stable and final_hold_stable)

    # Compute elevation difference between start and end
    h_start = float(np.mean(h[: max(1, int(n * 0.15))]))
    h_end = float(np.mean(h[int(n * 0.85) :]))
    delta_h = h_end - h_start

    press_info = elevation_to_hydrostatic_pressure(delta_h)

    return {
        "delta_h_m": delta_h,
        "delta_p_mmhg": press_info["delta_p_mmhg"],
        "velocity_m_s": v,
        "elevation_m": h,
        "hold_stable": hold_stable,
    }


class HydrostaticEngine:
    """Engine for hydrostatic pressure perturbations and IMU elevation tracking."""

    def __init__(
        self,
        rho_blood: float = RHO_BLOOD,
        g_gravity: float = G_GRAVITY,
    ) -> None:
        self.rho = float(rho_blood)
        self.g = float(g_gravity)
        self.pa_to_mmhg = PA_TO_MMHG
        self.mmhg_per_m = self.rho * self.g * self.pa_to_mmhg

    def calculate_pressure_shift(self, delta_h_m: float) -> Dict[str, float]:
        """Calculate Delta P_hydro in Pa and mmHg for a given height displacement."""
        p_pa = self.rho * self.g * float(delta_h_m)
        p_mmhg = p_pa * self.pa_to_mmhg
        return {
            "delta_h_m": float(delta_h_m),
            "delta_p_pa": float(p_pa),
            "delta_p_mmhg": float(p_mmhg),
        }

    def process_imu_stream(
        self,
        time_sec: np.ndarray,
        accel_vertical: np.ndarray,
        stationary_threshold: float = 0.06,
    ) -> Dict[str, Union[float, np.ndarray, bool]]:
        """Process vertical accelerometer time series with ZUPT drift correction."""
        return integrate_zupt_elevation(
            time_sec=time_sec,
            accel_vertical=accel_vertical,
            stationary_threshold=stationary_threshold,
        )
