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


class OrthogonalPhysiologicalExtractor:
    """
    Computes the strictly non-redundant 14-dimensional physiological vector.
    Excludes collinear multi-level widths (W10, W25, W50, W75).
    """

    FEATURE_NAMES = [
        "1_ptt_foot_ms",          # Foot-to-foot transit time [ms]
        "2_ptt_vmax_ms",          # Peak velocity transit time [ms]
        "3_epwv_mps",             # Estimated Pulse Wave Velocity [m/s]
        "4_hydrostatic_mmhg",     # Hydrostatic column correction rho*g*dh [mmHg]
        "5_hr_bpm",               # Heart Rate [BPM]
        "6_aix_pct",              # Augmentation Index AIx [%]
        "7_stiffness_index_si",   # Large-artery Stiffness Index Height / dT_DVP [m/s]
        "8_reflection_index_b_a", # Reflection / APG b/a compliance ratio
        "9_aging_index_agi",      # Vascular Aging Index (b - c - d - e) / a
        "10_ipa_area_ratio",      # Diastolic-to-Systolic Inflection Point Area ratio
        "11_crest_time_ratio",    # Systolic Crest Time ratio T_rise / T_cycle
        "12_decay_time_s",        # Diastolic runoff time / Windkessel RC decay [s]
        "13_snr_db",              # Spectral harmonic Signal-to-Noise Ratio [dB]
        "14_path_length_m",       # Arterial path length L = 0.286 * Height [m] (+ demo norm)
    ]

    def __init__(self, fs: int = 250):
        self.fs = fs

    def compute_spectral_snr_db(self, ppg: np.ndarray, hr_hz: float) -> float:
        import scipy.signal as signal
        nperseg = min(len(ppg), self.fs * 4)
        if nperseg < 16:
            return 12.0
        freqs, psd = signal.welch(ppg, fs=self.fs, nperseg=nperseg)
        band_mask = (freqs >= 0.65) & (freqs <= 8.0)
        f_band = freqs[band_mask]
        p_band = psd[band_mask]
        if len(p_band) == 0:
            return 0.0

        harm_mask = np.zeros_like(f_band, dtype=bool)
        for h in [1, 2, 3]:
            harm_mask |= np.abs(f_band - h * hr_hz) <= 0.18

        p_sig = float(np.sum(p_band[harm_mask]))
        p_noise = float(np.sum(p_band[~harm_mask])) + 1e-9
        return float(10.0 * np.log10(max(p_sig, 1e-9) / p_noise))

    def extract_14d_vector(
        self,
        reg_output: Dict[str, Any],
        palm_ppg_full: np.ndarray,
        height_m: float = 1.72,
        delta_h_m: float = 0.15,
        age: float = 40.0,
        bmi: float = 23.5,
    ) -> Optional[np.ndarray]:
        pairs = reg_output.get("certified_pairs", [])
        if len(pairs) == 0:
            return None

        # 1 & 2. PTT Foot-to-Foot and Peak-to-Peak (Vmax)
        ptt_foot_ms = float(np.median([p["ptt_foot_ms"] for p in pairs]))
        ptt_vmax_ms = float(np.median([p["ptt_vmax_ms"] for p in pairs]))

        # 14 & 3. Height-derived differential arterial path L = 0.286 * Height & ePWV
        L_diff_m = 0.286 * height_m
        epwv_mps = L_diff_m / (max(ptt_foot_ms, 1e-3) / 1000.0)

        # 4. Hydrostatic Blood Pressure Offset: rho * g * delta_h (Pa -> mmHg)
        rho_blood = 1060.0  # kg/m^3
        g_acc = 9.80665     # m/s^2
        pa_to_mmhg = 0.00750062
        hydro_mmhg = -rho_blood * g_acc * delta_h_m * pa_to_mmhg

        # 5. Heart Rate (BPM)
        t_cycles = np.array([p["t_cycle_sec"] for p in pairs])
        mean_t_cycle = float(np.mean(t_cycles)) if len(t_cycles) > 0 else 0.8
        hr_bpm = 60.0 / max(mean_t_cycle, 0.2)

        # Aggregate morphology metrics across certified beats
        aix_list, si_list, ba_list, agi_list, ipa_list, ct_list, dt_list = [], [], [], [], [], [], []
        for p in pairs:
            beat = p["palm_beat_shifted_norm"]
            apg = p["apg_waves"]
            L_b = len(beat)
            sys_idx = int(np.argmax(beat[: max(3, int(0.45 * L_b))]))
            e_idx = min(L_b - 2, max(sys_idx + 2, apg.get("e_idx", L_b // 2)))

            dia_idx = e_idx + int(np.argmax(beat[e_idx:])) if e_idx < L_b - 2 else e_idx
            dt_dvp = max((dia_idx - sys_idx) / self.fs, 0.08)

            p1 = float(beat[sys_idx])
            p2 = float(beat[dia_idx])
            aix_list.append(((p2 - p1) / (p1 + 1e-8)) * 100.0)
            si_list.append(height_m / dt_dvp)
            a_val = max(apg.get("a", 1.0), 1e-6)
            ba_list.append(apg.get("b", -0.5) / a_val)
            agi_list.append((apg.get("b", 0) - apg.get("c", 0) - apg.get("d", 0) - apg.get("e", 0)) / a_val)

            trapz_func = np.trapezoid if hasattr(np, "trapezoid") else getattr(np, "trapz", None)
            area_sys = float(trapz_func(beat[:e_idx])) + 1e-6
            area_dia = float(trapz_func(beat[e_idx:]))
            ipa_list.append(area_dia / area_sys)

            t_rise = sys_idx / self.fs
            ct_list.append(t_rise / max(p["t_cycle_sec"], 0.1))
            dt_list.append(max(0.0, p["t_cycle_sec"] - t_rise))

        snr_db = self.compute_spectral_snr_db(palm_ppg_full, hr_bpm / 60.0)
        demo_conditioned_path = L_diff_m * (1.0 + 0.002 * (age - 40.0) + 0.001 * (bmi - 22.0))

        feat_14 = np.array([
            ptt_foot_ms,
            ptt_vmax_ms,
            epwv_mps,
            hydro_mmhg,
            hr_bpm,
            float(np.median(aix_list)) if aix_list else 0.0,
            float(np.median(si_list)) if si_list else 8.0,
            float(np.median(ba_list)) if ba_list else -0.5,
            float(np.median(agi_list)) if agi_list else -0.3,
            float(np.median(ipa_list)) if ipa_list else 0.6,
            float(np.median(ct_list)) if ct_list else 0.25,
            float(np.median(dt_list)) if dt_list else 0.55,
            snr_db,
            demo_conditioned_path,
        ], dtype=np.float32)

        return feat_14

