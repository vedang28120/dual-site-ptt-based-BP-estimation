"""
End-to-End Mobile Blood Pressure Inference Pipeline.

Orchestrates the complete edge processing chain on Xiaomi Poco M6 Pro 5G / SM4450:
1. Ingests raw asynchronous dual camera streams tagged with SENSOR_TIMESTAMP.
2. Synchronizes streams onto a 125 Hz uniform master time-grid via PCHIP interpolation.
3. Solves the HRVP-PINN hemodynamic physics engine:
   - Sub-frame intersecting tangent foot detector (<= 0.75 ms precision).
   - Anthropometric path length & ePWV.
   - IMU hydrostatic perturbation (Delta P = rho * g * Delta h).
   - Traube-Hering-Mayer 0.2 Hz respiratory compliance d(ePWV)/dP.
   - Closed-form arterial stiffness (E0, alpha) and baseline MAP.
   - Standardized 6D physiological anchor vector z_HRVP.
4. Normalizes wave tensors into (1, 1, 300) contact morphology and facial timing sequences.
5. Ingests tensors into TFLiteRunner for deep learning regression.
6. Fuses analytical HRVP baseline with neural residual predictions.
7. Enforces clinical plausibility guards, capillary blanching detection, motion artifact
   rejection, and sustained 30 FPS timing verification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import torch

from src.optical.time_grid_synchronizer import (
    TimeGridSynchronizer,
    SynchronizationResult,
    compute_timestamp_jitter,
)
from src.optical.crop_extractor import compute_perfusion_index
from src.pinn.stiffness_solver import solve_hrvp
from src.mobile.tflite_runner import TFLiteRunner


@dataclass
class PipelineConfig:
    """Configuration parameters for the mobile BP estimation pipeline."""
    target_fs_hz: float = 125.0
    seq_len: int = 300
    nominal_fps: float = 30.0
    fusion_weight: float = 0.50          # Weight given to neural vs analytical prediction
    blanching_pi_threshold: float = 0.15 # Perfusion Index floor below which blanching is flagged (%)
    saturation_dc_threshold: float = 245.0
    underexposure_dc_threshold: float = 30.0
    min_snr_db: float = 4.0              # SNR floor for motion artifact detection
    sbp_min_mmhg: float = 60.0           # Clinical lower bound
    sbp_max_mmhg: float = 220.0          # Clinical upper bound
    dbp_min_mmhg: float = 30.0           # Clinical lower bound
    dbp_max_mmhg: float = 140.0          # Clinical upper bound
    min_pulse_pressure_mmhg: float = 15.0# Minimum physiological PP
    epwv_min_mps: float = 3.0
    epwv_max_mps: float = 18.0


@dataclass
class SignalQualityMetrics:
    """Signal quality indices and hardware integrity indicators."""
    perfusion_index: float
    dc_mean_finger: float
    ac_ptp_finger: float
    snr_face_db: float
    snr_finger_db: float
    front_jitter_std_ms: float
    rear_jitter_std_ms: float
    initial_phase_offset_ms: float
    dropped_frames_front: int
    dropped_frames_rear: int
    sustained_30fps: bool
    blanching_detected: bool
    motion_artifact_detected: bool
    dropped_frames_detected: bool


@dataclass
class PipelineResult:
    """Output container for end-to-end mobile blood pressure estimation."""
    sbp: float                           # Final Systolic Blood Pressure (mmHg)
    dbp: float                           # Final Diastolic Blood Pressure (mmHg)
    p_map: float                         # Final Mean Arterial Pressure (mmHg)
    epwv_mps: float                      # Estimated Pulse Wave Velocity (m/s)
    status: str                          # Clinical acquisition status code
    is_valid: bool                       # True if status == 'VALID_ACQUISITION'
    sbp_analytical: float                # Closed-form HRVP analytical SBP
    dbp_analytical: float                # Closed-form HRVP analytical DBP
    map_analytical: float                # Closed-form HRVP analytical MAP
    sbp_neural: float                    # DS-TCN neural regression SBP
    dbp_neural: float                    # DS-TCN neural regression DBP
    delta_ptt_ms: float                  # Sub-frame pulse transit time (ms)
    alpha: float                         # Hughes arterial stiffness coefficient (mmHg^-1)
    e0_pa: float                         # Moens-Korteweg zero-pressure elastic modulus (Pa)
    compliance_slope: float              # d(ePWV)/dP compliance slope
    anchor_vector: np.ndarray            # 6D HRVP physiological anchor vector
    quality: SignalQualityMetrics        # Optical and temporal signal quality
    total_latency_ms: float              # Total end-to-end processing latency
    inference_latency_ms: float          # Neural model inference latency

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to flat dictionary for serialization and evaluation."""
        d = asdict(self)
        d["anchor_vector"] = self.anchor_vector.tolist()
        return d

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"PipelineResult has no field '{key}'")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def estimate_snr_db(signal: np.ndarray) -> float:
    """
    Estimates optical signal-to-noise ratio in decibels.
    Compares pulsatile signal variance against high-frequency differenced noise.
    """
    if len(signal) < 4:
        return 0.0
    sig = np.asarray(signal, dtype=np.float64)
    sig_power = float(np.var(sig))
    diff = np.diff(sig)
    noise_power = float(np.var(diff) / 2.0) + 1e-12
    return float(10.0 * np.log10(max(sig_power, 1e-12) / noise_power))


def normalize_waveform_tensor(signal: np.ndarray, target_len: int = 300) -> np.ndarray:
    """
    Resamples and normalizes optical waveforms into exact (1, 1, target_len) tensor.
    
    Applies shape-preserving linear interpolation and min-max normalization to [0, 1].
    """
    sig = np.asarray(signal, dtype=np.float32)
    n = len(sig)
    if n == target_len:
        resampled = sig.copy()
    else:
        # Uniform resampling to target length
        x_orig = np.linspace(0.0, 1.0, n)
        x_target = np.linspace(0.0, 1.0, target_len)
        resampled = np.interp(x_target, x_orig, sig).astype(np.float32)

    min_v = float(np.min(resampled))
    max_v = float(np.max(resampled))
    val_range = max_v - min_v

    if val_range > 1e-6:
        normalized = (resampled - min_v) / val_range
    else:
        normalized = np.zeros_like(resampled)

    return normalized.reshape(1, 1, target_len)


def _extract_pulsatile_channel(arr: np.ndarray) -> np.ndarray:
    """Extract pulsatile channel (green index 1 if multi-channel, else 0 or 1D)."""
    if arr.ndim > 1:
        if arr.shape[1] > 1:
            return arr[:, 1]
        return arr[:, 0]
    return arr


class DualSiteBPPipeline:
    """
    Master Mobile Blood Pressure Inference Pipeline.
    
    Integrates:
    - Android Camera2 nanosecond SENSOR_TIMESTAMP PCHIP synchronization.
    - HRVP-PINN closed-form hemodynamic parameter solver.
    - Static-buffered INT8 TFLite runner for DS-TCN edge inference.
    - Physiological bounds, blanching, motion artifact, and dropped frame guards.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        tflite_runner: Optional[TFLiteRunner] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the pipeline.

        Parameters
        ----------
        config : Optional[PipelineConfig]
            Configuration parameters.
        tflite_runner : Optional[TFLiteRunner]
            Existing TFLiteRunner instance.
        model_path : Optional[str]
            Path to .tflite INT8 flatbuffer if creating a new runner.
        """
        self.config = config or PipelineConfig()
        self.synchronizer = TimeGridSynchronizer(target_fs_hz=self.config.target_fs_hz)

        if tflite_runner is not None:
            self.tflite_runner = tflite_runner
        else:
            self.tflite_runner = TFLiteRunner(
                model_path=model_path,
                seq_len=self.config.seq_len,
            )

    def process_session(self, session: Dict[str, Any]) -> PipelineResult:
        """
        Processes a full acquisition session dictionary.
        
        Expected session keys:
          - 't_front_ns': np.ndarray shape (N,)
          - 'face_rgb': np.ndarray shape (N, 3) or (N, C)
          - 't_rear_ns': np.ndarray shape (M,)
          - 'finger_rgb': np.ndarray shape (M, 3)
          - 'imu_data': dict with {"delta_h_m": float, "hold_stable": bool}
          - 'user_profile': dict with {"height_m": float, "arm_span_m": Optional[float]}
          - 'nominal_fps': optional float (default 30.0)

        Returns
        -------
        PipelineResult
        """
        t_front_ns = session["t_front_ns"]
        face_rgb = session["face_rgb"]
        t_rear_ns = session["t_rear_ns"]
        finger_rgb = session["finger_rgb"]
        imu_data = session.get("imu_data", {"delta_h_m": 0.0, "hold_stable": True})
        user_profile = session.get("user_profile", {"height_m": 1.75})
        nominal_fps = float(session.get("nominal_fps", self.config.nominal_fps))

        return self.process_streams(
            t_front_ns=t_front_ns,
            face_rgb=face_rgb,
            t_rear_ns=t_rear_ns,
            finger_rgb=finger_rgb,
            imu_data=imu_data,
            user_profile=user_profile,
            nominal_fps=nominal_fps,
        )

    def process_streams(
        self,
        t_front_ns: np.ndarray,
        face_rgb: np.ndarray,
        t_rear_ns: np.ndarray,
        finger_rgb: np.ndarray,
        imu_data: Dict[str, Any],
        user_profile: Dict[str, Any],
        nominal_fps: float = 30.0,
    ) -> PipelineResult:
        """
        Executes end-to-end estimation from raw timestamped optical and IMU streams.
        """
        t_start_pipeline = time.perf_counter()

        # ---------------------------------------------------------------------
        # 1. Master Time-Grid PCHIP Resampling (Eliminates Jitter & Phase Offset)
        # ---------------------------------------------------------------------
        # Check and sanitize raw inputs for non-finite values (NaN / Inf)
        face_arr = np.asarray(face_rgb, dtype=np.float32)
        finger_arr = np.asarray(finger_rgb, dtype=np.float32)
        nan_corrupted = (not np.all(np.isfinite(face_arr))) or (not np.all(np.isfinite(finger_arr)))

        if nan_corrupted:
            clean_face_rgb = np.nan_to_num(face_arr, nan=130.0, posinf=255.0, neginf=0.0)
            clean_finger_rgb = np.nan_to_num(finger_arr, nan=150.0, posinf=255.0, neginf=0.0)
        else:
            clean_face_rgb = face_arr
            clean_finger_rgb = finger_arr

        # Raw optical SNR checks prior to spline smoothing
        raw_face_sig = _extract_pulsatile_channel(clean_face_rgb)
        raw_finger_sig = _extract_pulsatile_channel(clean_finger_rgb)
        snr_face_raw = estimate_snr_db(raw_face_sig)
        snr_finger_raw = estimate_snr_db(raw_finger_sig)

        sync_result: SynchronizationResult = self.synchronizer.synchronize_streams(
            t_front_ns=t_front_ns,
            face_rgb=clean_face_rgb,
            t_rear_ns=t_rear_ns,
            finger_rgb=clean_finger_rgb,
            nominal_fps=nominal_fps,
        )

        resampled_face = sync_result.resampled_face
        resampled_finger = sync_result.resampled_finger

        # Green channel extracts primary pulsatile optical absorption
        # (Forehead rPPG index 1, Contact finger PPG center crop index 1)
        face_green = _extract_pulsatile_channel(resampled_face)
        finger_green = _extract_pulsatile_channel(resampled_finger)

        # ---------------------------------------------------------------------
        # 2. Optical Signal Quality & Artifact Checks
        # ---------------------------------------------------------------------
        # Analyze frame arrival jitter and detect dropped frames
        jitter_front = compute_timestamp_jitter(t_front_ns, nominal_fps)
        jitter_rear = compute_timestamp_jitter(t_rear_ns, nominal_fps)
        dropped_front = int(jitter_front["dropped_frames"])
        dropped_rear = int(jitter_rear["dropped_frames"])
        total_dropped = dropped_front + dropped_rear

        # Contact PPG Perfusion Index and Blanching check
        dc_mean_finger = float(np.mean(finger_green))
        ac_ptp_finger = float(np.ptp(finger_green))
        pi = compute_perfusion_index(ac_ptp_finger, dc_mean_finger)

        blanching_detected = (
            pi < self.config.blanching_pi_threshold
            or dc_mean_finger >= self.config.saturation_dc_threshold
            or dc_mean_finger <= self.config.underexposure_dc_threshold
        )

        # Motion Artifact & Noise checks across both raw and resampled representations
        snr_face_resampled = estimate_snr_db(face_green)
        snr_finger_resampled = estimate_snr_db(finger_green)

        snr_face = min(snr_face_raw, snr_face_resampled)
        snr_finger = min(snr_finger_raw, snr_finger_resampled)

        motion_artifact_detected = (
            nan_corrupted
            or snr_face < self.config.min_snr_db
            or snr_finger < self.config.min_snr_db
            or np.isnan(face_green).any()
            or np.isnan(finger_green).any()
        )

        # 10-second sustained 30 FPS timing check
        duration_sec = sync_result.duration_sec
        sustained_30fps = (total_dropped == 0 and duration_sec >= 9.5)

        quality_metrics = SignalQualityMetrics(
            perfusion_index=float(round(pi, 3)),
            dc_mean_finger=float(round(dc_mean_finger, 2)),
            ac_ptp_finger=float(round(ac_ptp_finger, 2)),
            snr_face_db=float(round(snr_face, 2)),
            snr_finger_db=float(round(snr_finger, 2)),
            front_jitter_std_ms=float(round(sync_result.front_jitter_std_ms, 3)),
            rear_jitter_std_ms=float(round(sync_result.rear_jitter_std_ms, 3)),
            initial_phase_offset_ms=float(round(sync_result.initial_phase_offset_ms, 3)),
            dropped_frames_front=dropped_front,
            dropped_frames_rear=dropped_rear,
            sustained_30fps=bool(sustained_30fps),
            blanching_detected=bool(blanching_detected),
            motion_artifact_detected=bool(motion_artifact_detected),
            dropped_frames_detected=bool(total_dropped > 0),
        )

        # ---------------------------------------------------------------------
        # 3. HRVP-PINN Hemodynamic Engine
        # ---------------------------------------------------------------------
        hrvp_output = solve_hrvp(
            face_signal=face_green,
            finger_signal=finger_green,
            imu_data=imu_data,
            user_profile=user_profile,
            fs=self.config.target_fs_hz,
        )

        sbp_analytical = float(hrvp_output["sbp_analytical_mmhg"])
        dbp_analytical = float(hrvp_output["dbp_analytical_mmhg"])
        map_analytical = float(hrvp_output["map_analytical_mmhg"])
        epwv_est = float(hrvp_output["epwv_mps"])
        delta_ptt_ms = float(hrvp_output["delta_ptt_ms"])
        alpha_est = float(hrvp_output["alpha"])
        e0_est = float(hrvp_output["e0_pa"])
        compliance_slope = float(hrvp_output["compliance_slope"])
        anchor_vec = np.asarray(hrvp_output["anchor_vector"], dtype=np.float32)

        # ---------------------------------------------------------------------
        # 4. Wave Tensor Normalization for DS-TCN
        # ---------------------------------------------------------------------
        x_morph = normalize_waveform_tensor(finger_green, target_len=self.config.seq_len)
        x_timing = normalize_waveform_tensor(face_green, target_len=self.config.seq_len)

        # ---------------------------------------------------------------------
        # 5. TFLite INT8 Neural Inference
        # ---------------------------------------------------------------------
        t_infer_start = time.perf_counter()
        sbp_neural, dbp_neural = self.tflite_runner.predict(
            morph_input=x_morph,
            timing_input=x_timing,
            anchor_input=anchor_vec,
        )
        inference_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

        # Handle any NaNs from corrupted inputs
        if np.isnan(sbp_neural) or np.isinf(sbp_neural):
            sbp_neural = sbp_analytical
        if np.isnan(dbp_neural) or np.isinf(dbp_neural):
            dbp_neural = dbp_analytical

        # ---------------------------------------------------------------------
        # 6. Physiological Baseline & Neural Residual Fusion
        # ---------------------------------------------------------------------
        # If severe artifact detected, weight toward analytical hemodynamic model
        if blanching_detected or motion_artifact_detected:
            effective_weight = 0.05
        else:
            effective_weight = self.config.fusion_weight

        sbp_fused = (1.0 - effective_weight) * sbp_analytical + effective_weight * sbp_neural
        dbp_fused = (1.0 - effective_weight) * dbp_analytical + effective_weight * dbp_neural

        # ---------------------------------------------------------------------
        # 7. Clinical Plausibility Guards
        # ---------------------------------------------------------------------
        clamped = False

        # Guard: SBP / DBP physiologically valid limits
        if sbp_fused < self.config.sbp_min_mmhg:
            sbp_fused = self.config.sbp_min_mmhg
            clamped = True
        elif sbp_fused > self.config.sbp_max_mmhg:
            sbp_fused = self.config.sbp_max_mmhg
            clamped = True

        if dbp_fused < self.config.dbp_min_mmhg:
            dbp_fused = self.config.dbp_min_mmhg
            clamped = True
        elif dbp_fused > self.config.dbp_max_mmhg:
            dbp_fused = self.config.dbp_max_mmhg
            clamped = True

        # Guard: Minimum pulse pressure (SBP must exceed DBP by min_pulse_pressure)
        if sbp_fused - dbp_fused < self.config.min_pulse_pressure_mmhg:
            sbp_fused = min(self.config.sbp_max_mmhg, dbp_fused + self.config.min_pulse_pressure_mmhg + 5.0)
            clamped = True

        # Guard: Mean Arterial Pressure consistency
        p_map = dbp_fused + (1.0 / 3.0) * (sbp_fused - dbp_fused)

        # Determine clinical status code
        if blanching_detected:
            status = "BLANCHING_DETECTED"
            is_valid = False
        elif motion_artifact_detected:
            status = "MOTION_ARTIFACT"
            is_valid = False
        elif total_dropped > 0:
            status = "DROPPED_FRAMES_DETECTED"
            is_valid = False
        elif clamped:
            status = "PHYSIOLOGICAL_CLAMP_APPLIED"
            is_valid = True
        else:
            status = "VALID_ACQUISITION"
            is_valid = True

        total_latency_ms = (time.perf_counter() - t_start_pipeline) * 1000.0

        return PipelineResult(
            sbp=float(round(sbp_fused, 1)),
            dbp=float(round(dbp_fused, 1)),
            p_map=float(round(p_map, 1)),
            epwv_mps=float(round(epwv_est, 2)),
            status=status,
            is_valid=is_valid,
            sbp_analytical=float(round(sbp_analytical, 1)),
            dbp_analytical=float(round(dbp_analytical, 1)),
            map_analytical=float(round(map_analytical, 1)),
            sbp_neural=float(round(sbp_neural, 1)),
            dbp_neural=float(round(dbp_neural, 1)),
            delta_ptt_ms=float(round(delta_ptt_ms, 2)),
            alpha=float(alpha_est),
            e0_pa=float(e0_est),
            compliance_slope=float(compliance_slope),
            anchor_vector=anchor_vec,
            quality=quality_metrics,
            total_latency_ms=float(round(total_latency_ms, 2)),
            inference_latency_ms=float(round(inference_latency_ms, 2)),
        )
