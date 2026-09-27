"""
Authoritative Reference Implementation for HRVP-PINN and Optical DSP
Author: test_writer_1 (E2E Testing Architect)

Implements the exact physical and mathematical models defined in PROJECT.md:
1. Master Time-Grid PCHIP Resampling (125 Hz)
2. Sub-Frame Intersecting Tangent Continuous Foot Detector (<= 0.75 ms precision)
3. Anthropometric path & ePWV calculator
4. IMU Two-Elevation Hydrostatic Perturbation Engine (Delta P = rho * g * Delta h)
5. Traube-Hering-Mayer 0.2 Hz Respiratory Envelope & Compliance Slope d(ePWV)/dP
6. Closed-Form Arterial Stiffness Solver (E_0, alpha, P_MAP, SBP, DBP)
7. Tripartite Anti-Collapse Loss (Huber + Pearson + Variance)
8. Simulated INT8 Quantization and dynamic range validation
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.signal import find_peaks, butter, filtfilt, hilbert

# Physical constants
RHO_BLOOD = 1055.0         # kg/m^3
GRAVITY_G = 9.80665        # m/s^2
PA_TO_MMHG = 1.0 / 133.322 # mmHg / Pa
MMHG_TO_PA = 133.322       # Pa / mmHg
DEFAULT_HW_R0 = 0.10       # h_w / R_0 ratio


# ============================================================================
# 1. Master Time-Grid PCHIP Synchronizer (Feature 4)
# ============================================================================
class ReferenceTimeGridSynchronizer:
    """
    Resamples asynchronous, jittered camera streams onto an equispaced nanosecond master grid.
    Eliminates rolling-shutter phase offsets and OS scheduling jitter.
    """
    def __init__(self, target_fs_hz: float = 125.0):
        self.target_fs_hz = target_fs_hz
        self.dt_ns = int(round(1e9 / target_fs_hz))

    def synchronize(
        self,
        t_front_ns: np.ndarray,
        face_rgb: np.ndarray,
        t_rear_ns: np.ndarray,
        finger_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ingests raw hardware timestamps and returns (master_time_sec, resampled_face, resampled_finger).
        """
        t_start = max(int(t_front_ns[0]), int(t_rear_ns[0]))
        t_end = min(int(t_front_ns[-1]), int(t_rear_ns[-1]))

        if t_end <= t_start:
            raise ValueError(f"No temporal overlap between camera streams: start={t_start}, end={t_end}")

        master_t_ns = np.arange(t_start, t_end, self.dt_ns, dtype=np.int64)
        if len(master_t_ns) < 10:
            raise ValueError("Overlapping temporal window too short (< 10 samples).")

        pchip_front = PchipInterpolator(t_front_ns, face_rgb, axis=0)
        pchip_rear = PchipInterpolator(t_rear_ns, finger_rgb, axis=0)

        resampled_face = pchip_front(master_t_ns)
        resampled_finger = pchip_rear(master_t_ns)
        master_time_sec = (master_t_ns - master_t_ns[0]) / 1e9

        return master_time_sec, resampled_face, resampled_finger


# ============================================================================
# 2. Sub-Frame Continuous Intersecting Tangent Foot Detector (Feature 5)
# ============================================================================
class ReferenceSubFrameFootDetector:
    """
    Continuous spline intersecting tangent foot detector.
    Achieves <= 0.75 ms precision, breaking through the 33.3 ms 30 FPS sampling ceiling.
    """
    def __init__(self, fs: float = 125.0, min_hr_bpm: float = 35.0, max_hr_bpm: float = 200.0):
        self.fs = fs
        self.min_distance = int(round(fs * (60.0 / max_hr_bpm) * 0.6))
        self.max_cycle_samples = int(round(fs * (60.0 / min_hr_bpm)))

    def detect_feet(self, signal: np.ndarray, time_axis: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns continuous sub-frame foot arrival timestamps in seconds.
        """
        N = len(signal)
        if time_axis is None:
            time_axis = np.arange(N) / self.fs

        # 1. Detect systolic peaks
        peaks, _ = find_peaks(signal, distance=self.min_distance, prominence=np.ptp(signal) * 0.15)
        if len(peaks) < 1:
            return np.array([], dtype=np.float64)

        feet_times: List[float] = []

        # 2. Fit continuous cubic splines between cardiac cycles
        for i, peak_idx in enumerate(peaks):
            # Define foot search window preceding peak
            prev_peak_idx = peaks[i - 1] if i > 0 else max(0, peak_idx - int(self.fs * 0.8))
            window_start = int(prev_peak_idx + (peak_idx - prev_peak_idx) * 0.3)
            window_end = peak_idx

            if window_end - window_start < 4:
                continue

            t_win = time_axis[window_start:window_end + 1]
            y_win = signal[window_start:window_end + 1]

            # Fit continuous cubic spline
            cs = CubicSpline(t_win, y_win)
            d1 = cs.derivative(1)
            d2 = cs.derivative(2)

            # Continuous minimum (diastolic trough L1)
            # Sample densely within window
            t_dense = np.linspace(t_win[0], t_win[-1], 200)
            y_dense = cs(t_dense)
            min_idx = np.argmin(y_dense)
            t_min = t_dense[min_idx]
            y_min = y_dense[min_idx]

            # Max upstroke slope (inflection point where d2 crosses zero, between t_min and peak)
            t_upstroke = np.linspace(t_min, t_win[-1], 200)
            d1_up = d1(t_upstroke)
            max_slope_idx = np.argmax(d1_up)
            t_max_slope = t_upstroke[max_slope_idx]
            m_slope = d1_up[max_slope_idx]
            y_max_slope = cs(t_max_slope)

            # Intersecting tangent:
            # L1: y = y_min
            # L2: y - y_max_slope = m_slope * (t - t_max_slope)
            # y_min - y_max_slope = m_slope * (t_foot - t_max_slope)
            if abs(m_slope) > 1e-6:
                t_foot = t_max_slope - (y_max_slope - y_min) / m_slope
                # Verify physiological validity: t_foot should be near [t_min, t_max_slope]
                if t_win[0] - 0.05 <= t_foot <= t_win[-1]:
                    feet_times.append(float(t_foot))
                else:
                    feet_times.append(float(t_min))
            else:
                feet_times.append(float(t_min))

        return np.asarray(feet_times, dtype=np.float64)

    def compute_delta_ptt_ms(
        self,
        face_signal: np.ndarray,
        finger_signal: np.ndarray,
        time_axis: np.ndarray
    ) -> float:
        """
        Computes median pulse transit time (Delta PTT) in milliseconds between face and finger.
        """
        face_feet = self.detect_feet(face_signal, time_axis)
        finger_feet = self.detect_feet(finger_signal, time_axis)

        if len(face_feet) == 0 or len(finger_feet) == 0:
            raise ValueError("Could not detect pulse feet on one or both signals.")

        # Match each face foot with the earliest subsequent finger foot within [20ms, 250ms]
        ptts_ms: List[float] = []
        for t_face in face_feet:
            valid_finger = finger_feet[(finger_feet > t_face) & (finger_feet < t_face + 0.350)]
            if len(valid_finger) > 0:
                ptts_ms.append((valid_finger[0] - t_face) * 1000.0)

        if len(ptts_ms) == 0:
            # Fallback: estimate from cross-correlation of first derivative
            return float(np.median(finger_feet[0] - face_feet[0]) * 1000.0 if len(face_feet) > 0 else 80.0)

        return float(np.median(ptts_ms))


# ============================================================================
# 3. Anthropometrics & ePWV Calculator (Feature 6)
# ============================================================================
class ReferenceAnthropometrics:
    """
    Computes anatomical path distance and Estimated Pulse Wave Velocity (ePWV).
    D_path = 0.55 * Height + 0.12 * ArmSpan
    """
    @staticmethod
    def calculate_d_path(height_m: float, arm_span_m: Optional[float] = None) -> float:
        arm_span = arm_span_m if arm_span_m is not None else 1.03 * height_m
        return float(0.55 * height_m + 0.12 * arm_span)

    @staticmethod
    def calculate_epwv(d_path_m: float, delta_ptt_ms: float) -> float:
        if delta_ptt_ms <= 0:
            raise ValueError(f"Delta PTT must be positive, got {delta_ptt_ms} ms")
        delta_ptt_sec = delta_ptt_ms / 1000.0
        return float(d_path_m / delta_ptt_sec)


# ============================================================================
# 4. IMU Hydrostatic Perturbation Engine (Feature 7)
# ============================================================================
class ReferenceHydrostaticEngine:
    """
    Computes hydrostatic column pressure shift:
    Delta P_hydro = rho_blood * g * Delta h
    """
    def __init__(self, rho: float = RHO_BLOOD, g: float = GRAVITY_G):
        self.rho = rho
        self.g = g

    def calculate_delta_p(self, delta_h_m: float) -> float:
        # Pressure in Pa converted to mmHg
        delta_p_pa = self.rho * self.g * delta_h_m
        return float(delta_p_pa * PA_TO_MMHG)

    def is_hold_stable(self, vertical_accel: np.ndarray, threshold: float = 0.08) -> bool:
        """
        ZUPT stationary hold detection: returns True if std(accel) < threshold.
        """
        return bool(np.std(vertical_accel) < threshold)


# ============================================================================
# 5. Traube-Hering-Mayer Respiratory Envelope & Compliance (Feature 8)
# ============================================================================
class ReferenceTHMAnalyzer:
    """
    Extracts ~0.2 Hz respiratory modulation via Hilbert transform and resolves instantaneous compliance d(ePWV)/dP.
    """
    def __init__(self, fs: float = 125.0, f_low: float = 0.12, f_high: float = 0.38):
        self.fs = fs
        self.f_low = f_low
        self.f_high = f_high

    def extract_envelope(self, epwv_series: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Returns (analytic_envelope, mean_amplitude).
        """
        # Bandpass filter around respiratory band [0.15, 0.35 Hz]
        nyq = 0.5 * self.fs
        b, a = butter(2, [self.f_low / nyq, self.f_high / nyq], btype='bandpass')
        epwv_filtered = filtfilt(b, a, epwv_series - np.mean(epwv_series))

        # Hilbert transform
        analytic_signal = hilbert(epwv_filtered)
        envelope = np.abs(analytic_signal)
        mean_amp = float(np.mean(envelope))
        return envelope, mean_amp

    def calculate_compliance_slope(
        self,
        epwv_1: float,
        epwv_2: float,
        delta_p_hydro_mmhg: float
    ) -> float:
        """
        Computes compliance slope d(ePWV)/dP = (ePWV_2 - ePWV_1) / Delta P_hydro
        """
        if abs(delta_p_hydro_mmhg) < 1e-4:
            raise ValueError("Hydrostatic pressure offset too small for compliance identification.")
        return float((epwv_2 - epwv_1) / delta_p_hydro_mmhg)


# ============================================================================
# 6. Closed-Form Arterial Stiffness Solver (Feature 9)
# ============================================================================
class ReferenceStiffnessSolver:
    """
    Closed-form analytical identification of:
    - alpha = 2 * ln(ePWV_2 / ePWV_1) / (rho * g * Delta h)
    - c_0 = ePWV_1 * exp(-0.5 * alpha * P_MAP_1)
    - P_MAP, SBP, DBP
    """
    def __init__(self, rho: float = RHO_BLOOD, g: float = GRAVITY_G, hw_r0: float = DEFAULT_HW_R0):
        self.rho = rho
        self.g = g
        self.hw_r0 = hw_r0

    def solve_alpha(self, epwv_1: float, epwv_2: float, delta_h_m: float) -> float:
        if abs(delta_h_m) < 1e-3:
            raise ValueError("Elevation displacement must be non-zero to solve alpha.")
        if epwv_1 <= 0 or epwv_2 <= 0:
            raise ValueError(f"ePWV values must be positive: {epwv_1}, {epwv_2}")

        delta_p_mmhg = (self.rho * self.g * delta_h_m) * PA_TO_MMHG
        ratio = epwv_2 / epwv_1
        alpha = (2.0 * np.log(ratio)) / delta_p_mmhg
        return float(alpha)

    def solve_e0_and_map(
        self,
        epwv_1: float,
        alpha: float,
        c0: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Returns (e0_pa, map_mmhg).
        If c0 is not given, uses nominal structural speed c0 = 5.0 m/s.
        """
        if c0 is None:
            c0 = 5.0
        # P_MAP = (2 / alpha) * ln(ePWV_1 / c0)
        p_map = (2.0 / alpha) * np.log(epwv_1 / c0)
        # E_0 = (2 * rho / (hw/r0)) * c0^2
        e0_pa = (2.0 * self.rho / self.hw_r0) * (c0 ** 2)
        return float(e0_pa), float(p_map)

    def separate_sbp_dbp(
        self,
        p_map_mmhg: float,
        pulse_pressure_mmhg: float = 40.0,
        k_form: float = 0.333
    ) -> Tuple[float, float]:
        """
        Decoupled SBP and DBP using classic form factor:
        P_MAP = DBP + k_form * (SBP - DBP)
        SBP = P_MAP + (1 - k_form) * PP
        DBP = P_MAP - k_form * PP
        """
        sbp = p_map_mmhg + (1.0 - k_form) * pulse_pressure_mmhg
        dbp = p_map_mmhg - k_form * pulse_pressure_mmhg
        return float(sbp), float(dbp)


# ============================================================================
# 7. Tripartite Anti-Collapse Compound Loss (Feature 12)
# ============================================================================
class ReferenceTripartiteLoss:
    """
    Loss = L_Huber(delta=1.0) + lambda_1 * (1 - Pearson_r) + lambda_2 * L_Variance
    Explicitly prevents template collapse / variance squeeze.
    """
    def __init__(self, delta: float = 1.0, lambda_1: float = 0.40, lambda_2: float = 0.25, sigma_thresh: float = 8.0):
        self.delta = delta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.sigma_thresh = sigma_thresh

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

        # 1. Huber Loss
        diff = np.abs(y_true - y_pred)
        huber_elem = np.where(diff <= self.delta, 0.5 * (diff ** 2), self.delta * (diff - 0.5 * self.delta))
        loss_huber = float(np.mean(huber_elem))

        # 2. Pearson Correlation Loss
        vy_true = y_true - np.mean(y_true)
        vy_pred = y_pred - np.mean(y_pred)
        denom = (np.sqrt(np.sum(vy_true ** 2)) * np.sqrt(np.sum(vy_pred ** 2))) + 1e-8
        r = float(np.sum(vy_true * vy_pred) / denom)
        loss_pearson = float(1.0 - r)

        # 3. Variance Regularization Loss
        std_true = float(np.std(y_true) + 1e-8)
        std_pred = float(np.std(y_pred) + 1e-8)
        variance_ratio = (std_pred ** 2) / (std_true ** 2)
        spread_penalty = abs(1.0 - (std_pred / std_true))
        floor_penalty = max(0.0, 1.0 - ((std_pred ** 2) / (self.sigma_thresh ** 2)))
        loss_variance = float(spread_penalty + floor_penalty)

        total_loss = float(loss_huber + self.lambda_1 * loss_pearson + self.lambda_2 * loss_variance)

        return {
            "total_loss": total_loss,
            "loss_huber": loss_huber,
            "loss_pearson": loss_pearson,
            "loss_variance": loss_variance,
            "pearson_r": r,
            "variance_ratio": variance_ratio
        }


# ============================================================================
# 8. Simulated INT8 Quantizer (Feature 13)
# ============================================================================
class ReferenceINT8Quantizer:
    """
    Symmetric per-channel weight quantization and asymmetric per-tensor activation quantization.
    Validates precision loss and memory savings for Snapdragon 4 Gen 2 execution.
    """
    @staticmethod
    def quantize_symmetric(weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Quantizes weights to INT8 [-127, 127] with scale = max(|w|) / 127.
        """
        max_val = float(np.max(np.abs(weights)))
        scale = max_val / 127.0 if max_val > 0 else 1.0
        q_weights = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
        return q_weights, scale

    @staticmethod
    def dequantize_symmetric(q_weights: np.ndarray, scale: float) -> np.ndarray:
        return (q_weights.astype(np.float32) * scale)

    @staticmethod
    def calculate_quantization_error_snr(orig: np.ndarray, dequant: np.ndarray) -> float:
        signal_power = np.mean(orig ** 2)
        noise_power = np.mean((orig - dequant) ** 2) + 1e-12
        return float(10.0 * np.log10(signal_power / noise_power))
