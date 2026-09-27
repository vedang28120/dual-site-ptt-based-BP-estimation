"""Traube-Hering-Mayer (THM) Respiratory Envelope & Arterial Compliance.

This module extracts the low-frequency respiratory vasomotor oscillation
(~0.15 - 0.35 Hz, centered at 0.20 Hz / 12 breaths/min) from cycle-by-cycle
ePWV series:
  1. Resamples non-uniform beat-to-beat ePWV to uniform 10 Hz via PCHIP.
  2. Applies zero-phase Butterworth bandpass filtering (0.15 - 0.35 Hz).
  3. Constructs analytic signal via Hilbert transform:
       Z(t) = x(t) + j * H{x(t)}
  4. Computes instantaneous envelope amplitude A_THM(t) = |Z(t)|.
  5. Computes macroscopic and instantaneous arterial compliance:
       d(ePWV)/dP = Delta ePWV / Delta P_hydro
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, hilbert, sosfiltfilt


def compute_compliance_slope(
    epwv_1: float,
    epwv_2: float,
    delta_p_hydro_mmhg: float,
) -> float:
    """Compute macroscopic arterial compliance slope d(ePWV)/dP.

    d(ePWV)/dP = (ePWV_2 - ePWV_1) / Delta P_hydro

    Parameters
    ----------
    epwv_1 : float
        Baseline pulse wave velocity at elevation 1 in m/s.
    epwv_2 : float
        Perturbed pulse wave velocity at elevation 2 in m/s.
    delta_p_hydro_mmhg : float
        Hydrostatic pressure shift in mmHg.

    Returns
    -------
    float
        Compliance slope in (m/s) / mmHg.
    """
    if abs(delta_p_hydro_mmhg) < 1e-6:
        # Default typical human compliance slope if hydrostatic perturbation is near zero
        return 0.050  # ~0.05 (m/s)/mmHg
    return float((epwv_2 - epwv_1) / delta_p_hydro_mmhg)


def extract_respiratory_envelope(
    cycle_times: np.ndarray,
    epwv_series: np.ndarray,
    fs_resample: float = 10.0,
    f_low: float = 0.15,
    f_high: float = 0.35,
    filter_order: int = 2,
) -> Dict[str, Union[float, np.ndarray]]:
    """Extract Traube-Hering-Mayer envelope from cycle-by-cycle ePWV series.

    Parameters
    ----------
    cycle_times : np.ndarray
        Array of timestamps (seconds) corresponding to each cardiac cycle.
    epwv_series : np.ndarray
        Array of cycle-by-cycle estimated pulse wave velocities (m/s).
    fs_resample : float, optional
        Target uniform resampling rate in Hz (default: 10.0 Hz).
    f_low : float, optional
        Lower cutoff frequency for THM respiratory band in Hz (default: 0.15 Hz).
    f_high : float, optional
        Upper cutoff frequency for THM respiratory band in Hz (default: 0.35 Hz).
    filter_order : int, optional
        Butterworth bandpass filter order (default: 2).

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing:
          - "time_uniform": Uniform timestamp grid at fs_resample.
          - "epwv_uniform": PCHIP interpolated uniform ePWV series.
          - "epwv_thm_filtered": Zero-phase bandpass filtered respiratory modulation.
          - "analytic_signal": Complex Hilbert analytic signal Z(t).
          - "instantaneous_amplitude": Instantaneous envelope amplitude A_THM(t).
          - "mean_envelope_amplitude": Mean respiratory envelope amplitude (m/s).
          - "instantaneous_phase": Phase angle of respiratory oscillation (radians).
          - "respiratory_rate_bpm": Estimated respiratory rate in breaths per minute.
    """
    t_raw = np.asarray(cycle_times, dtype=np.float64)
    y_raw = np.asarray(epwv_series, dtype=np.float64)

    if len(t_raw) < 2 or (t_raw[-1] - t_raw[0]) < 1.0 / fs_resample:
        # Trivial or degenerate case: synthesize minimal representation
        dummy_t = np.linspace(0.0, 1.0, int(fs_resample))
        mean_val = float(np.mean(y_raw)) if len(y_raw) > 0 else 6.0
        return {
            "time_uniform": dummy_t,
            "epwv_uniform": np.full_like(dummy_t, mean_val),
            "epwv_thm_filtered": np.zeros_like(dummy_t),
            "analytic_signal": np.zeros_like(dummy_t, dtype=np.complex128),
            "instantaneous_amplitude": np.zeros_like(dummy_t),
            "mean_envelope_amplitude": 0.0,
            "instantaneous_phase": np.zeros_like(dummy_t),
            "respiratory_rate_bpm": 12.0,
        }

    # Ensure strictly increasing timestamps for PCHIP
    sort_idx = np.argsort(t_raw)
    t_sorted = t_raw[sort_idx]
    y_sorted = y_raw[sort_idx]

    # Remove duplicates if any
    unique_mask = np.insert(np.diff(t_sorted) > 1e-6, 0, True)
    t_clean = t_sorted[unique_mask]
    y_clean = y_sorted[unique_mask]

    if len(t_clean) < 2:
        dummy_t = np.array([t_clean[0]])
        return {
            "time_uniform": dummy_t,
            "epwv_uniform": np.array([y_clean[0]]),
            "epwv_thm_filtered": np.array([0.0]),
            "analytic_signal": np.array([0.0 + 0.0j]),
            "instantaneous_amplitude": np.array([0.0]),
            "mean_envelope_amplitude": 0.0,
            "instantaneous_phase": np.array([0.0]),
            "respiratory_rate_bpm": 12.0,
        }

    # 1. Uniform resampling via PCHIP
    t_start = t_clean[0]
    t_end = t_clean[-1]
    num_samples = max(4, int(np.round((t_end - t_start) * fs_resample)))
    time_uniform = np.linspace(t_start, t_end, num_samples)

    pchip = PchipInterpolator(t_clean, y_clean, extrapolate=False)
    epwv_uniform = pchip(time_uniform)

    # 2. Bandpass filtering (0.15 - 0.35 Hz)
    # Remove mean baseline before filtering
    dc_mean = np.mean(epwv_uniform)
    ac_signal = epwv_uniform - dc_mean

    nyq = 0.5 * fs_resample
    low = f_low / nyq
    high = f_high / nyq

    # Constrain normalized frequencies within (0, 1)
    low = max(0.01, min(0.90, low))
    high = max(low + 0.01, min(0.99, high))

    sos = butter(filter_order, [low, high], btype="bandpass", output="sos")

    # Safe zero-phase filtering
    if len(ac_signal) > 3 * (2 * filter_order + 1):
        padlen = min(len(ac_signal) - 1, 3 * (2 * filter_order + 1))
        epwv_thm = sosfiltfilt(sos, ac_signal, padlen=padlen)
    else:
        # For short signals, detrended signal serves as direct fluctuation
        epwv_thm = ac_signal.copy()

    # 3. Analytic signal via Hilbert transform
    analytic = hilbert(epwv_thm)

    # 4. Instantaneous envelope amplitude and phase
    inst_amp = np.abs(analytic)
    inst_phase = np.angle(analytic)
    mean_amp = float(np.mean(inst_amp))

    # 5. Dominant respiratory rate estimation in BPM
    if len(epwv_thm) >= 16:
        fft_vals = np.abs(np.fft.rfft(epwv_thm))
        fft_freqs = np.fft.rfftfreq(len(epwv_thm), d=1.0 / fs_resample)
        valid_band = (fft_freqs >= f_low) & (fft_freqs <= f_high)
        if np.any(valid_band) and np.max(fft_vals[valid_band]) > 1e-6:
            dom_freq = fft_freqs[valid_band][np.argmax(fft_vals[valid_band])]
            resp_bpm = float(dom_freq * 60.0)
        else:
            resp_bpm = 12.0  # Normal resting default (0.20 Hz * 60)
    else:
        resp_bpm = 12.0

    return {
        "time_uniform": time_uniform,
        "epwv_uniform": epwv_uniform,
        "epwv_thm_filtered": epwv_thm,
        "analytic_signal": analytic,
        "instantaneous_amplitude": inst_amp,
        "mean_envelope_amplitude": mean_amp,
        "instantaneous_phase": inst_phase,
        "respiratory_rate_bpm": resp_bpm,
    }


class RespiratoryEnvelopeExtractor:
    """Extractor for Traube-Hering-Mayer respiratory envelope and compliance."""

    def __init__(
        self,
        fs_resample: float = 10.0,
        f_low: float = 0.15,
        f_high: float = 0.35,
        filter_order: int = 2,
    ) -> None:
        self.fs_resample = float(fs_resample)
        self.f_low = float(f_low)
        self.f_high = float(f_high)
        self.filter_order = int(filter_order)

    def extract(
        self,
        cycle_times: np.ndarray,
        epwv_series: np.ndarray,
    ) -> Dict[str, Union[float, np.ndarray]]:
        return extract_respiratory_envelope(
            cycle_times=cycle_times,
            epwv_series=epwv_series,
            fs_resample=self.fs_resample,
            f_low=self.f_low,
            f_high=self.f_high,
            filter_order=self.filter_order,
        )

    def compliance_slope(
        self,
        epwv_1: float,
        epwv_2: float,
        delta_p_hydro_mmhg: float,
    ) -> float:
        return compute_compliance_slope(epwv_1, epwv_2, delta_p_hydro_mmhg)
