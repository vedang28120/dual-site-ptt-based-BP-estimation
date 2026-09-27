"""
Synthetic Physiological Waveform Generator with Closed-Form Ground Truth
Author: test_writer_1 (E2E Testing Architect)

Generates dual-stream optical pulses (proximal face rPPG + distal finger contact PPG),
IMU elevation kinematics, Traube-Hering-Mayer respiratory envelope modulation (~0.2 Hz),
variable frame rate (VFR) timestamp jitter, and Hughes non-linear arterial stiffness parameters
for independent, opaque-box E2E testing and template collapse auditing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import hilbert, butter, filtfilt

# Standard physical constants
RHO_BLOOD = 1055.0         # Blood density in kg/m^3
GRAVITY_G = 9.80665        # Gravitational acceleration in m/s^2
PA_TO_MMHG = 1.0 / 133.322 # Conversion factor Pa -> mmHg
MMHG_TO_PA = 133.322       # Conversion factor mmHg -> Pa
DEFAULT_HW_R0_RATIO = 0.10 # Standard wall-thickness to radius ratio (h_w / R_0)


@dataclass
class PhysiologicalProfile:
    """Anthropometric and hemodynamic parameters defining a clinical subject."""
    height_m: float = 1.75
    arm_span_m: Optional[float] = None
    sbp_mmhg: float = 120.0
    dbp_mmhg: float = 80.0
    hr_bpm: float = 72.0
    resp_rate_bpm: float = 12.0
    alpha: float = 0.024         # Hughes stiffness coefficient in mmHg^-1
    e0_pa: float = 0.85e5        # Zero-pressure elastic modulus in Pa
    delta_h_m: float = 0.0       # IMU elevation offset in meters (+ = raised, - = lowered)
    snr_db: float = 30.0         # Signal to noise ratio in dB
    vfr_jitter_ms: float = 2.0   # Inter-frame timestamp jitter standard deviation
    phase_offset_ms: float = 8.5 # Front vs rear camera rolling-shutter phase offset


def calculate_d_path(height_m: float, arm_span_m: Optional[float] = None) -> float:
    """
    Computes differential arterial travel distance:
    D_path = D_arm - D_carotid = 0.55 * Height + 0.12 * ArmSpan.
    If arm_span_m is None, uses biomechanical ratio ArmSpan = 1.03 * Height.
    """
    effective_arm_span = arm_span_m if arm_span_m is not None else 1.03 * height_m
    d_path = 0.55 * height_m + 0.12 * effective_arm_span
    return float(d_path)


def calculate_c0(e0_pa: float, hw_r0_ratio: float = DEFAULT_HW_R0_RATIO, rho: float = RHO_BLOOD) -> float:
    """
    Computes baseline Moens-Korteweg structural wave speed:
    c_0 = sqrt( (h_w * E_0) / (2 * rho * R_0) )
    """
    c0 = np.sqrt((hw_r0_ratio * e0_pa) / (2.0 * rho))
    return float(c0)


def calculate_epwv(c0: float, alpha: float, pressure_mmhg: Any) -> Any:
    """
    Computes estimated Pulse Wave Velocity (ePWV) via Hughes non-linear elasticity:
    PWV(P) = c_0 * exp( (alpha / 2) * P )
    """
    res = c0 * np.exp(0.5 * alpha * pressure_mmhg)
    if np.isscalar(res) or (isinstance(res, np.ndarray) and res.ndim == 0):
        return float(res)
    return res


def calculate_delta_ptt_ms(d_path_m: float, epwv_mps: float) -> float:
    """
    Computes pulse transit time in milliseconds:
    Delta PTT = (D_path / ePWV) * 1000
    """
    return float((d_path_m / epwv_mps) * 1000.0)


def calculate_hydrostatic_delta_p(delta_h_m: float, rho: float = RHO_BLOOD, g: float = GRAVITY_G) -> float:
    """
    Computes hydrostatic column pressure change in mmHg:
    Delta P_hydro = (rho * g * Delta h) / 133.322
    """
    delta_p_pa = rho * g * delta_h_m
    return float(delta_p_pa * PA_TO_MMHG)


def calculate_compliance_slope(epwv_mps: float, alpha: float) -> float:
    """
    Computes derivative of ePWV with respect to pressure:
    d(ePWV)/dP = (alpha / 2) * ePWV [in (m/s) / mmHg]
    """
    return float(0.5 * alpha * epwv_mps)


def generate_single_cardiac_cycle(
    phase: np.ndarray,
    is_distal: bool = True,
    damping_factor: float = 1.0
) -> np.ndarray:
    """
    Synthesizes a single normalized cardiac cycle [0, 1] using asymmetric multi-Gaussian profile.
    Distal contact PPG retains crisp dicrotic notch and diastolic wave.
    Proximal facial rPPG experiences Windkessel microvascular low-pass attenuation.
    """
    # Phase in [0, 1]
    # Systolic upstroke peak
    mu1 = 0.16
    sig1 = 0.055
    systolic = np.exp(-((phase - mu1) ** 2) / (2 * sig1 ** 2))

    # Dicrotic reflection notch
    mu2 = 0.40
    sig2 = 0.075
    notch_amp = 0.45 if is_distal else (0.12 * damping_factor)
    dicrotic = notch_amp * np.exp(-((phase - mu2) ** 2) / (2 * sig2 ** 2))

    # Diastolic runoff decay
    decay = 0.15 * np.exp(-3.0 * np.maximum(0.0, phase - mu2))

    pulse = systolic + dicrotic + decay
    pulse = (pulse - np.min(pulse)) / (np.max(pulse) - np.min(pulse) + 1e-12)
    return pulse


class SyntheticPhysiologicalGenerator:
    """
    High-fidelity physiological waveform generator with closed-form ground truth.
    Simulates dual optical cameras, IMU elevation, respiratory modulation, and noise.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def generate_session(
        self,
        profile: PhysiologicalProfile,
        duration_sec: float = 10.0,
        nominal_camera_fps: float = 30.0,
        master_fs_hz: float = 125.0
    ) -> Dict[str, Any]:
        """
        Synthesizes a complete 10-second dual-camera acquisition session.
        Returns hardware timestamps, raw RGB streams, IMU kinematics, and ground truth parameters.
        """
        # 1. Anthropometrics & Anatomical Path
        d_path = calculate_d_path(profile.height_m, profile.arm_span_m)
        d_carotid = 0.18 * profile.height_m
        d_arm = d_carotid + d_path

        # 2. Hemodynamics and Hydrostatics
        p_map_baseline = profile.dbp_mmhg + (1.0 / 3.0) * (profile.sbp_mmhg - profile.dbp_mmhg)
        delta_p_hydro = calculate_hydrostatic_delta_p(profile.delta_h_m)
        p_map_perturbed = p_map_baseline + delta_p_hydro
        sbp_perturbed = profile.sbp_mmhg + delta_p_hydro
        dbp_perturbed = profile.dbp_mmhg + delta_p_hydro

        c0 = calculate_c0(profile.e0_pa)
        epwv_baseline = calculate_epwv(c0, profile.alpha, p_map_baseline)
        epwv_perturbed = calculate_epwv(c0, profile.alpha, p_map_perturbed)
        delta_ptt_baseline_ms = calculate_delta_ptt_ms(d_path, epwv_baseline)
        delta_ptt_perturbed_ms = calculate_delta_ptt_ms(d_path, epwv_perturbed)
        compliance_slope = calculate_compliance_slope(epwv_perturbed, profile.alpha)

        # 3. Time Base & Respiratory Traube-Hering-Mayer Modulation
        # Micro-second resolution simulation grid (1000 Hz) for smooth wave propagation
        sim_dt = 0.001
        t_sim = np.arange(0.0, duration_sec, sim_dt)
        f_resp = profile.resp_rate_bpm / 60.0 # ~0.2 Hz
        thm_amp_mmhg = 6.0                     # Typical 6 mmHg respiratory oscillation
        p_thm_modulation = thm_amp_mmhg * np.sin(2.0 * np.pi * f_resp * t_sim)

        # Continuous ePWV and Delta PTT trajectory
        epwv_continuous = calculate_epwv(c0, profile.alpha, p_map_perturbed + p_thm_modulation)
        t_delay_face = d_carotid / epwv_continuous
        t_delay_finger = d_arm / epwv_continuous
        ptt_continuous_ms = (t_delay_finger - t_delay_face) * 1000.0

        # 4. Continuous Pulse Synthesis
        t_cardiac = 60.0 / profile.hr_bpm
        phase_master = (t_sim % t_cardiac) / t_cardiac

        # Proximal face wave (shifted by t_delay_face, microvascular damping = 0.30)
        face_phase = ((t_sim - t_delay_face) % t_cardiac) / t_cardiac
        wave_face_ideal = generate_single_cardiac_cycle(face_phase, is_distal=False, damping_factor=0.30)

        # Distal finger wave (shifted by t_delay_finger, sharp dicrotic notch)
        finger_phase = ((t_sim - t_delay_finger) % t_cardiac) / t_cardiac
        wave_finger_ideal = generate_single_cardiac_cycle(finger_phase, is_distal=True)

        # 5. Camera Sampling & Variable Frame Rate (VFR) Timestamps
        # Front camera: starts at t0 = 0.0, nominal dt = 1/fps + jitter
        n_frames_expected = int(np.ceil(duration_sec * nominal_camera_fps))
        nominal_dt_sec = 1.0 / nominal_camera_fps

        # Front timestamps with jitter
        jitter_front = self.rng.normal(0.0, profile.vfr_jitter_ms * 1e-3, size=n_frames_expected)
        t_front_sec = np.cumsum(np.maximum(0.015, nominal_dt_sec + jitter_front))
        t_front_sec = t_front_sec - t_front_sec[0] # align first frame to 0
        t_front_sec = t_front_sec[t_front_sec < duration_sec]
        t_front_ns = (t_front_sec * 1e9).astype(np.int64)

        # Rear camera: starts with rolling-shutter phase offset + jitter
        phase_offset_sec = profile.phase_offset_ms * 1e-3
        jitter_rear = self.rng.normal(0.0, profile.vfr_jitter_ms * 1e-3, size=n_frames_expected)
        t_rear_sec = np.cumsum(np.maximum(0.015, nominal_dt_sec + jitter_rear))
        t_rear_sec = t_rear_sec - t_rear_sec[0] + phase_offset_sec
        t_rear_sec = t_rear_sec[t_rear_sec < duration_sec]
        t_rear_ns = (t_rear_sec * 1e9).astype(np.int64)

        # Interpolate ideal signals onto camera discrete timestamps
        face_pchip = PchipInterpolator(t_sim, wave_face_ideal)
        finger_pchip = PchipInterpolator(t_sim, wave_finger_ideal)

        sampled_face = face_pchip(t_front_sec)
        sampled_finger = finger_pchip(t_rear_sec)

        # 6. Add Sensor Noise and Optical Modulation
        # Contact PPG: High DC baseline (e.g. 150 out of 255), AC amplitude ~ 25
        # Face rPPG: Forehead DC ~ 130, AC amplitude ~ 2.0 (attenuated)
        noise_face_std = (1.0 / (10.0 ** (profile.snr_db / 20.0)))
        noise_finger_std = (1.0 / (10.0 ** ((profile.snr_db + 10.0) / 20.0)))

        noise_face = self.rng.normal(0.0, noise_face_std, size=len(t_front_sec))
        noise_finger = self.rng.normal(0.0, noise_finger_std, size=len(t_rear_sec))

        face_ac_dc = 130.0 + 2.5 * (sampled_face - 0.5) + noise_face * 2.5
        finger_ac_dc = 155.0 + 35.0 * (sampled_finger - 0.5) + noise_finger * 35.0

        # Construct RGB tensors
        # Front face RGB: [R, G, B] with green carrying rPPG modulation
        face_rgb = np.zeros((len(t_front_sec), 3), dtype=np.float32)
        face_rgb[:, 0] = 145.0 + 1.0 * (sampled_face - 0.5)
        face_rgb[:, 1] = face_ac_dc.astype(np.float32)
        face_rgb[:, 2] = 115.0 + 0.8 * (sampled_face - 0.5)

        # Rear finger RGB: 64x64 center crop spatial average
        finger_rgb = np.zeros((len(t_rear_sec), 3), dtype=np.float32)
        finger_rgb[:, 0] = 200.0 + 10.0 * (sampled_finger - 0.5)
        finger_rgb[:, 1] = finger_ac_dc.astype(np.float32)
        finger_rgb[:, 2] = 80.0 + 5.0 * (sampled_finger - 0.5)

        # 7. IMU Kinematic Stream (Delta h tracking)
        imu_data = {
            "delta_h_m": float(profile.delta_h_m),
            "hold_stable": bool(profile.delta_h_m != 0.0 or duration_sec >= 3.0),
            "delta_p_hydro_mmhg": float(delta_p_hydro),
            "elevation_profile_m": np.full(len(t_sim), profile.delta_h_m, dtype=np.float32)
        }

        # 8. User Profile Dict for Interface Contract
        user_profile = {
            "height_m": float(profile.height_m),
            "arm_span_m": float(profile.arm_span_m if profile.arm_span_m else 1.03 * profile.height_m),
            "e0_pa": float(profile.e0_pa),
            "c0": float(c0),
        }

        # 9. Master Ground Truth Metadata
        ground_truth = {
            "sbp_baseline_mmhg": float(profile.sbp_mmhg),
            "dbp_baseline_mmhg": float(profile.dbp_mmhg),
            "map_baseline_mmhg": float(p_map_baseline),
            "sbp_perturbed_mmhg": float(sbp_perturbed),
            "dbp_perturbed_mmhg": float(dbp_perturbed),
            "map_perturbed_mmhg": float(p_map_perturbed),
            "hr_bpm": float(profile.hr_bpm),
            "resp_rate_bpm": float(profile.resp_rate_bpm),
            "d_path_m": float(d_path),
            "c0_mps": float(c0),
            "alpha": float(profile.alpha),
            "e0_pa": float(profile.e0_pa),
            "epwv_baseline_mps": float(epwv_baseline),
            "epwv_perturbed_mps": float(epwv_perturbed),
            "delta_ptt_baseline_ms": float(delta_ptt_baseline_ms),
            "delta_ptt_perturbed_ms": float(delta_ptt_perturbed_ms),
            "compliance_slope": float(compliance_slope),
            "delta_p_hydro_mmhg": float(delta_p_hydro),
            "anchor_vector": np.array([
                epwv_perturbed,
                delta_ptt_perturbed_ms,
                delta_p_hydro,
                compliance_slope,
                profile.e0_pa,
                profile.alpha
            ], dtype=np.float32)
        }

        return {
            "t_front_ns": t_front_ns,
            "face_rgb": face_rgb,
            "t_rear_ns": t_rear_ns,
            "finger_rgb": finger_rgb,
            "imu_data": imu_data,
            "user_profile": user_profile,
            "ground_truth": ground_truth,
            "sim_master": {
                "t_sec": t_sim,
                "wave_face_ideal": wave_face_ideal,
                "wave_finger_ideal": wave_finger_ideal,
                "epwv_continuous": epwv_continuous,
                "ptt_continuous_ms": ptt_continuous_ms
            }
        }
