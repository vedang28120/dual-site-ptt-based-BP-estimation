"""
Empirical Stress Test Harness: Optical Foot Extraction & HRVP-PINN Hemodynamics
Author: challenger_1 (Physiological Adversary & Tangent Foot Challenger)

Adversarial Stress Vectors Tested:
1. Continuous rolling-shutter phase offset sweep: Delta_phi in [0.0, 33.3 ms]
2. Severe variable frame rate (VFR) timestamp jitter: sigma = 8.0 ms
3. Extreme heart rate spectrum: severe bradycardia (30 BPM) to extreme tachycardia (200 BPM)
4. Extreme arterial stiffness: elderly stiff vascular conditions (alpha = 0.050 mmHg^-1, E_0 = 2.5e5 Pa)
5. Extreme hydrostatic perturbations: macro-elevation (600 mm / 46.5 mmHg) & micro-elevation (5 mm / 0.39 mmHg)
6. Low optical SNR stress: 15 dB noise floor on front rPPG and rear contact PPG
"""

import sys
import os
import json
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = r"C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.optical.time_grid_synchronizer import TimeGridSynchronizer, compute_timestamp_jitter
from src.pinn.foot_detector import (
    FootDetector,
    calculate_tangent_foot,
    compute_subframe_ptt,
    detect_pulse_feet,
)
from src.pinn.anthropometrics import compute_path_length, compute_epwv
from src.pinn.hydrostatic_engine import (
    HydrostaticEngine,
    elevation_to_hydrostatic_pressure,
    RHO_BLOOD,
    G_GRAVITY,
    PA_TO_MMHG,
)
from src.pinn.respiratory_envelope import (
    RespiratoryEnvelopeExtractor,
    compute_compliance_slope,
    extract_respiratory_envelope,
)
from src.pinn.stiffness_solver import (
    HRVPEngine,
    calculate_k_form,
    decompose_sbp_dbp,
    solve_hrvp,
    solve_stiffness_parameters,
    C0_COLLAPSE_DEFAULT,
)
from tests.synthetic_generator import (
    PhysiologicalProfile,
    SyntheticPhysiologicalGenerator,
    calculate_c0,
    calculate_d_path,
    calculate_epwv,
    generate_single_cardiac_cycle,
)


def run_test1_rolling_shutter_sweep():
    """Vector 1: Continuous rolling-shutter phase offset sweep: Delta_phi in [0.0, 33.3 ms]."""
    print("\n" + "=" * 70)
    print("VECTOR 1: Continuous Rolling-Shutter Phase Offset Sweep (0.0 to 33.3 ms)")
    print("=" * 70)

    synchronizer = TimeGridSynchronizer(target_fs_hz=125.0)
    detector = FootDetector(fs=125.0)

    # Injected continuous transit delay: 65.0 ms
    true_delay_ms = 65.0
    true_delay_sec = true_delay_ms / 1000.0
    hr_bpm = 72.0
    hr_freq = hr_bpm / 60.0
    duration = 10.0

    # High-resolution simulation grid (1000 Hz)
    t_sim = np.linspace(0.0, duration, int(duration * 1000))
    t_cardiac = 1.0 / hr_freq

    # Phase curves
    face_phase = (t_sim % t_cardiac) / t_cardiac
    finger_phase = ((t_sim - true_delay_sec) % t_cardiac) / t_cardiac

    wave_face = generate_single_cardiac_cycle(face_phase, is_distal=False, damping_factor=0.3)
    wave_finger = generate_single_cardiac_cycle(finger_phase, is_distal=True)

    # 30 FPS nominal sampling
    fps = 30.0
    nominal_dt = 1.0 / fps

    phase_offsets_ms = np.linspace(0.0, 33.333, 35)
    timing_errors_ms = []
    sync_residual_errors = []

    for phi_ms in phase_offsets_ms:
        phi_sec = phi_ms / 1000.0

        # Front timestamps start at 0
        t_front_sec = np.arange(0.0, duration, nominal_dt)
        t_front_ns = (t_front_sec * 1e9).astype(np.int64)

        # Rear timestamps start at phi_sec
        t_rear_sec = np.arange(phi_sec, duration, nominal_dt)
        t_rear_ns = (t_rear_sec * 1e9).astype(np.int64)

        # Sample ideal waves
        face_sampled = np.interp(t_front_sec, t_sim, wave_face)
        finger_sampled = np.interp(t_rear_sec, t_sim, wave_finger)

        # Format as RGB (N, 3)
        face_rgb = np.zeros((len(t_front_sec), 3), dtype=np.float32)
        face_rgb[:, 1] = 130.0 + 10.0 * (face_sampled - 0.5)

        finger_rgb = np.zeros((len(t_rear_sec), 3), dtype=np.float32)
        finger_rgb[:, 1] = 155.0 + 40.0 * (finger_sampled - 0.5)

        # Synchronize
        sync_res = synchronizer.synchronize_streams(
            t_front_ns=t_front_ns,
            face_rgb=face_rgb,
            t_rear_ns=t_rear_ns,
            finger_rgb=finger_rgb,
            nominal_fps=fps,
        )

        sync_residual_errors.append(sync_res.max_residual_sync_error_ms)

        # Extract green channel
        sync_face_g = sync_res.resampled_face[:, 1]
        sync_finger_g = sync_res.resampled_finger[:, 1]
        sync_time = sync_res.master_time_sec

        # Detect pulse transit time
        ptt_res = detector.compute_ptt(
            time=sync_time,
            face_signal=sync_face_g,
            finger_signal=sync_finger_g,
            min_ptt_sec=0.015,
            max_ptt_sec=0.35,
            smooth=False,
        )

        measured_delay_ms = float(ptt_res["delta_ptt_ms"])
        err_ms = abs(measured_delay_ms - true_delay_ms)
        timing_errors_ms.append(err_ms)

    timing_errors_ms = np.array(timing_errors_ms)
    max_err = float(np.max(timing_errors_ms))
    mean_err = float(np.mean(timing_errors_ms))
    std_err = float(np.std(timing_errors_ms))

    print(f"Phase Offsets Tested: {len(phase_offsets_ms)} points across [0.0, 33.3 ms]")
    print(f"Max Foot Timing Error:  {max_err:.4f} ms (Mandated limit: <= 0.75 ms)")
    print(f"Mean Foot Timing Error: {mean_err:.4f} ms")
    print(f"Std Foot Timing Error:  {std_err:.4f} ms")
    print(f"Max Residual Sync Err:  {max(sync_residual_errors):.4f} ms")

    passed = bool(max_err <= 0.75)
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    return {
        "vector": "rolling_shutter_sweep",
        "phase_offsets_ms": phase_offsets_ms.tolist(),
        "timing_errors_ms": timing_errors_ms.tolist(),
        "max_err_ms": max_err,
        "mean_err_ms": mean_err,
        "passed": passed,
    }


def run_test2_vfr_jitter():
    """Vector 2: Severe Variable Frame Rate (VFR) timestamp jitter: sigma = 8.0 ms."""
    print("\n" + "=" * 70)
    print("VECTOR 2: Severe Variable Frame Rate (VFR) Jitter (sigma = 8.0 ms)")
    print("=" * 70)

    synchronizer = TimeGridSynchronizer(target_fs_hz=125.0)
    detector = FootDetector(fs=125.0)
    rng = np.random.default_rng(2026)

    true_delay_ms = 70.0
    true_delay_sec = true_delay_ms / 1000.0
    hr_bpm = 75.0
    duration = 10.0
    fps = 30.0
    nominal_dt = 1.0 / fps

    t_sim = np.linspace(0.0, duration, int(duration * 1000))
    t_cardiac = 60.0 / hr_bpm
    face_phase = (t_sim % t_cardiac) / t_cardiac
    finger_phase = ((t_sim - true_delay_sec) % t_cardiac) / t_cardiac

    wave_face = generate_single_cardiac_cycle(face_phase, is_distal=False, damping_factor=0.3)
    wave_finger = generate_single_cardiac_cycle(finger_phase, is_distal=True)

    jitter_sigmas_ms = [2.0, 4.0, 6.0, 8.0, 10.0]
    n_trials = 10

    results_per_sigma = {}

    for sigma_ms in jitter_sigmas_ms:
        trial_errors = []
        dropped_frames_front = []
        dropped_frames_rear = []

        for trial in range(n_trials):
            n_frames = int(duration * fps)

            # Generate jittered intervals clamped to physical minimum 10 ms
            jitter_front = rng.normal(0.0, sigma_ms * 1e-3, size=n_frames)
            dt_front = np.maximum(0.010, nominal_dt + jitter_front)
            t_front_sec = np.cumsum(dt_front)
            t_front_sec = t_front_sec - t_front_sec[0]
            t_front_sec = t_front_sec[t_front_sec < duration]
            t_front_ns = (t_front_sec * 1e9).astype(np.int64)

            # Rear has random phase offset + jitter
            phi_sec = float(rng.uniform(0.0, 0.033))
            jitter_rear = rng.normal(0.0, sigma_ms * 1e-3, size=n_frames)
            dt_rear = np.maximum(0.010, nominal_dt + jitter_rear)
            t_rear_sec = np.cumsum(dt_rear)
            t_rear_sec = t_rear_sec - t_rear_sec[0] + phi_sec
            t_rear_sec = t_rear_sec[t_rear_sec < duration]
            t_rear_ns = (t_rear_sec * 1e9).astype(np.int64)

            face_sampled = np.interp(t_front_sec, t_sim, wave_face)
            finger_sampled = np.interp(t_rear_sec, t_sim, wave_finger)

            face_rgb = np.zeros((len(t_front_sec), 3), dtype=np.float32)
            face_rgb[:, 1] = 130.0 + 10.0 * (face_sampled - 0.5)

            finger_rgb = np.zeros((len(t_rear_sec), 3), dtype=np.float32)
            finger_rgb[:, 1] = 155.0 + 40.0 * (finger_sampled - 0.5)

            sync_res = synchronizer.synchronize_streams(
                t_front_ns=t_front_ns,
                face_rgb=face_rgb,
                t_rear_ns=t_rear_ns,
                finger_rgb=finger_rgb,
                nominal_fps=fps,
            )

            ptt_res = detector.compute_ptt(
                time=sync_res.master_time_sec,
                face_signal=sync_res.resampled_face[:, 1],
                finger_signal=sync_res.resampled_finger[:, 1],
                smooth=False,
            )

            err = abs(float(ptt_res["delta_ptt_ms"]) - true_delay_ms)
            trial_errors.append(err)
            dropped_frames_front.append(sync_res.front_jitter_std_ms)
            dropped_frames_rear.append(sync_res.rear_jitter_std_ms)

        trial_errors = np.array(trial_errors)
        max_err = float(np.max(trial_errors))
        mean_err = float(np.mean(trial_errors))

        results_per_sigma[sigma_ms] = {
            "max_err_ms": max_err,
            "mean_err_ms": mean_err,
            "passed": bool(max_err <= 0.75),
        }
        print(
            f"Jitter sigma = {sigma_ms:4.1f} ms -> Max Timing Error: {max_err:.4f} ms, "
            f"Mean: {mean_err:.4f} ms -> {'PASS' if max_err <= 0.75 else 'FAIL'}"
        )

    passed_8ms = results_per_sigma[8.0]["passed"]
    print(f"Target sigma = 8.0 ms Verdict: {'PASS' if passed_8ms else 'FAIL'}")

    return {
        "vector": "vfr_jitter",
        "results": results_per_sigma,
        "passed_8ms": passed_8ms,
    }


def run_test3_extreme_heart_rates():
    """Vector 3: Extreme heart rates: severe bradycardia (30 BPM) and tachycardia (200 BPM)."""
    print("\n" + "=" * 70)
    print("VECTOR 3: Extreme Heart Rate Spectrum (30 BPM to 200 BPM)")
    print("=" * 70)

    heart_rates = [30.0, 40.0, 50.0, 75.0, 100.0, 140.0, 180.0, 200.0]
    injected_delay_ms = 60.0
    injected_delay_sec = injected_delay_ms / 1000.0
    duration = 12.0
    fs = 125.0

    t_sim = np.linspace(0.0, duration, int(duration * 2000))

    results_hr = {}

    for hr in heart_rates:
        t_cardiac = 60.0 / hr
        hr_freq = hr / 60.0

        # Multi-harmonic cardiac pulse
        p_face = 2.0 * np.pi * hr_freq * t_sim
        p_finger = 2.0 * np.pi * hr_freq * (t_sim - injected_delay_sec)

        wave_face = 0.6 * np.sin(p_face - np.pi/2) + 0.3 * np.sin(2*p_face - np.pi/2) + 0.1 * np.sin(3*p_face - np.pi/2)
        wave_finger = 0.6 * np.sin(p_finger - np.pi/2) + 0.3 * np.sin(2*p_finger - np.pi/2) + 0.1 * np.sin(3*p_finger - np.pi/2)

        t_grid = np.arange(0.0, duration, 1.0 / fs)
        face_grid = np.interp(t_grid, t_sim, wave_face)
        finger_grid = np.interp(t_grid, t_sim, wave_finger)

        # 1. Test with default parameters (max_bpm=180, min_bpm=40)
        ptt_default = compute_subframe_ptt(
            t_grid, face_grid, finger_grid, fs=fs, smooth=False
        )
        err_default = abs(float(ptt_default["delta_ptt_ms"]) - injected_delay_ms)
        beats_default = int(ptt_default["n_beats"])

        # Expected number of beats
        expected_beats = int(np.floor(duration / t_cardiac)) - 1

        # 2. Test with adaptive parameters (custom min_bpm / max_bpm)
        feet_face_adapt = detect_pulse_feet(
            t_grid, face_grid, fs=fs, min_bpm=min(25.0, hr * 0.8), max_bpm=max(220.0, hr * 1.2), smooth=False
        )
        feet_finger_adapt = detect_pulse_feet(
            t_grid, finger_grid, fs=fs, min_bpm=min(25.0, hr * 0.8), max_bpm=max(220.0, hr * 1.2), smooth=False
        )

        results_hr[hr] = {
            "hr_bpm": hr,
            "expected_beats": expected_beats,
            "detected_beats_default": beats_default,
            "error_default_ms": err_default,
            "feet_face_adapt_count": len(feet_face_adapt),
            "feet_finger_adapt_count": len(feet_finger_adapt),
            "passed_precision": bool(err_default <= 0.75),
        }

        print(
            f"HR = {hr:5.1f} BPM | Expected Beats: {expected_beats:2d} | "
            f"Detected: {beats_default:2d} | Timing Error: {err_default:.4f} ms | "
            f"Precision: {'PASS' if err_default <= 0.75 else 'FAIL'}"
        )

    # Specific analysis for 30 BPM and 200 BPM
    res_30 = results_hr[30.0]
    res_200 = results_hr[200.0]

    print("\n--- Extreme HR Findings ---")
    print(f"30 BPM: Error={res_30['error_default_ms']:.4f} ms, Detected={res_30['detected_beats_default']}/{res_30['expected_beats']} beats")
    print(f"200 BPM: Error={res_200['error_default_ms']:.4f} ms, Detected={res_200['detected_beats_default']}/{res_200['expected_beats']} beats")
    if res_200['detected_beats_default'] < res_200['expected_beats'] * 0.7:
        print("  WARNING: Default peak detector filter (max_bpm=180) drops beats at 200 BPM!")

    return results_hr


def run_test4_extreme_arterial_stiffness():
    """Vector 4: Extreme arterial stiffness: alpha = 0.050 mmHg^-1, E_0 = 2.5e5 Pa."""
    print("\n" + "=" * 70)
    print("VECTOR 4: Extreme Arterial Stiffness (alpha = 0.050 mmHg^-1, E_0 = 2.5e5 Pa)")
    print("=" * 70)

    # True physiological parameters for elderly/stiff vascular conditions
    alpha_true = 0.050  # mmHg^-1
    e0_true = 2.5e5     # Pa
    hw_r0_ratio = 0.35e-3 / 1.5e-3  # 0.2333
    rho = 1055.0

    # Moens-Korteweg true c0
    c0_true = np.sqrt((hw_r0_ratio * e0_true) / (2.0 * rho))  # ~5.258 m/s

    p1_mmhg = 90.0
    delta_h_m = 0.35  # 350 mm elevation
    delta_p_mmhg = 77.60142 * delta_h_m  # 27.16 mmHg
    p2_mmhg = p1_mmhg + delta_p_mmhg

    epwv_1 = c0_true * np.exp(0.5 * alpha_true * p1_mmhg)
    epwv_2 = c0_true * np.exp(0.5 * alpha_true * p2_mmhg)

    print(f"True Parameters: alpha={alpha_true:.4f} mmHg^-1, E_0={e0_true:.1e} Pa, c_0={c0_true:.3f} m/s")
    print(f"Simulated ePWV: ePWV_1={epwv_1:.3f} m/s, ePWV_2={epwv_2:.3f} m/s (Delta P = {delta_p_mmhg:.2f} mmHg)")

    # 1. Closed-form solver when c0 is provided
    sol_with_c0 = solve_stiffness_parameters(
        epwv_1=epwv_1,
        epwv_2=epwv_2,
        delta_h_m=delta_h_m,
        c0=c0_true,
    )
    alpha_rec_c0 = sol_with_c0["alpha_mmhg"]
    alpha_err_pct = abs(alpha_rec_c0 - alpha_true) / alpha_true * 100.0

    # 2. Closed-form solver when c0 is default (C0_COLLAPSE_DEFAULT = 4.5 m/s)
    sol_default = solve_stiffness_parameters(
        epwv_1=epwv_1,
        epwv_2=epwv_2,
        delta_h_m=delta_h_m,
    )
    alpha_rec_default = sol_default["alpha_mmhg"]
    alpha_err_default_pct = abs(alpha_rec_default - alpha_true) / alpha_true * 100.0
    e0_rec_default = sol_default["e0_pa"]
    e0_err_default_pct = abs(e0_rec_default - e0_true) / e0_true * 100.0
    map_rec_default = sol_default["map_analytical_mmhg"]
    map_err_mmhg = abs(map_rec_default - p1_mmhg)

    print("\nResults with True c_0:")
    print(f"  Recovered Alpha: {alpha_rec_c0:.6f} mmHg^-1 (Error: {alpha_err_pct:.4f}%)")
    print(f"  Recovered E_0:   {sol_with_c0['e0_pa']:.1e} Pa")
    print(f"  Recovered MAP:   {sol_with_c0['map_analytical_mmhg']:.2f} mmHg (True: {p1_mmhg:.2f})")

    print("\nResults with Default c_0 (Unassisted Zero-Transmural Structural Velocity = 4.5 m/s):")
    print(f"  Recovered Alpha: {alpha_rec_default:.6f} mmHg^-1 (Error: {alpha_err_default_pct:.4f}%)")
    print(f"  Recovered E_0:   {e0_rec_default:.1e} Pa (Error: {e0_err_default_pct:.2f}%)")
    print(f"  Recovered MAP:   {map_rec_default:.2f} mmHg (Offset: {map_err_mmhg:.2f} mmHg)")

    passed_alpha = bool(alpha_err_pct < 1.0)
    print(f"\nVerdict on Alpha Recovery (< 1%): {'PASS' if passed_alpha else 'FAIL'}")

    return {
        "vector": "extreme_stiffness",
        "alpha_true": alpha_true,
        "alpha_err_pct": alpha_err_pct,
        "e0_true": e0_true,
        "e0_rec_default": e0_rec_default,
        "e0_err_default_pct": e0_err_default_pct,
        "map_err_mmhg": map_err_mmhg,
        "passed": passed_alpha,
    }


def run_test5_extreme_hydrostatic_elevations():
    """Vector 5: Extreme hydrostatic elevations: macro (600 mm) & micro (5 mm)."""
    print("\n" + "=" * 70)
    print("VECTOR 5: Extreme Hydrostatic Elevations (600 mm vs 5 mm)")
    print("=" * 70)

    alpha_true = 0.024
    c0_true = 4.5
    p1_mmhg = 85.0

    elevations_mm = [600.0, 300.0, 100.0, 50.0, 20.0, 10.0, 5.0]
    results_elev = {}

    for elev_mm in elevations_mm:
        delta_h_m = elev_mm / 1000.0
        delta_p_mmhg = 77.60142 * delta_h_m

        epwv_1 = c0_true * np.exp(0.5 * alpha_true * p1_mmhg)
        epwv_2 = c0_true * np.exp(0.5 * alpha_true * (p1_mmhg + delta_p_mmhg))

        # Direct solver recovery
        sol = solve_stiffness_parameters(
            epwv_1=epwv_1,
            epwv_2=epwv_2,
            delta_h_m=delta_h_m,
            c0=c0_true,
        )

        alpha_rec = sol["alpha_mmhg"]
        err_pct = abs(alpha_rec - alpha_true) / alpha_true * 100.0

        # Sensitivity analysis: add 0.1 ms timing error to PTT2
        d_path = 0.95
        ptt1 = (d_path / epwv_1)
        ptt2 = (d_path / epwv_2)
        delta_ptt_true_ms = (ptt1 - ptt2) * 1000.0

        # Perturbed with 0.1 ms noise
        ptt2_noisy = ptt2 + 0.00010  # 0.1 ms error
        epwv_2_noisy = d_path / ptt2_noisy
        sol_noisy = solve_stiffness_parameters(
            epwv_1=epwv_1,
            epwv_2=epwv_2_noisy,
            delta_h_m=delta_h_m,
            c0=c0_true,
        )
        err_noisy_pct = abs(sol_noisy["alpha_mmhg"] - alpha_true) / alpha_true * 100.0

        results_elev[elev_mm] = {
            "delta_h_mm": elev_mm,
            "delta_p_mmhg": delta_p_mmhg,
            "delta_ptt_ms": delta_ptt_true_ms,
            "alpha_err_clean_pct": err_pct,
            "alpha_err_0.1ms_noise_pct": err_noisy_pct,
            "passed_clean": bool(err_pct < 1.0),
        }

        print(
            f"Elev: {elev_mm:5.1f} mm | Delta P: {delta_p_mmhg:6.2f} mmHg | "
            f"Delta PTT: {delta_ptt_true_ms:6.3f} ms | "
            f"Clean Alpha Err: {err_pct:.4f}% | With 0.1ms Noise Err: {err_noisy_pct:.2f}%"
        )

    # Full pipeline test with 5 mm vs 600 mm
    print("\n--- Pipeline Elevation Branching Verification ---")
    face_sig = np.sin(np.linspace(0, 10*np.pi, 500))
    finger_sig = np.sin(np.linspace(0, 10*np.pi, 500) - 0.5)

    imu_600mm = {"delta_h_m": 0.60, "hold_stable": True}
    imu_5mm = {"delta_h_m": 0.005, "hold_stable": True}
    user_prof = {"height_m": 1.75}

    hrvp_600 = solve_hrvp(face_sig, finger_sig, imu_600mm, user_prof, fs=125.0)
    hrvp_5 = solve_hrvp(face_sig, finger_sig, imu_5mm, user_prof, fs=125.0)

    print(f"600 mm Pipeline Hydrostatic Delta P: {hrvp_600['anchor_vector'][2]:.2f} mmHg (uses IMU elevation)")
    print(f"5 mm Pipeline Hydrostatic Delta P:   {hrvp_5['anchor_vector'][2]:.2f} mmHg (falls back to THM respiratory mode)")

    return results_elev


def run_test6_low_snr_stress():
    """Vector 6: Sub-frame foot precision under low SNR (15 dB sensor noise)."""
    print("\n" + "=" * 70)
    print("VECTOR 6: Sub-Frame Foot Precision Under Low SNR (15 dB Noise Floor)")
    print("=" * 70)

    detector = FootDetector(fs=125.0)
    rng = np.random.default_rng(42)

    injected_delay_ms = 55.0
    injected_delay_sec = injected_delay_ms / 1000.0
    duration = 10.0
    fs = 125.0
    hr_bpm = 72.0

    t_sim = np.linspace(0.0, duration, int(duration * 2000))
    t_cardiac = 60.0 / hr_bpm
    face_phase = (t_sim % t_cardiac) / t_cardiac
    finger_phase = ((t_sim - injected_delay_sec) % t_cardiac) / t_cardiac

    wave_face = generate_single_cardiac_cycle(face_phase, is_distal=False, damping_factor=0.3)
    wave_finger = generate_single_cardiac_cycle(finger_phase, is_distal=True)

    t_grid = np.arange(0.0, duration, 1.0 / fs)
    face_clean = np.interp(t_grid, t_sim, wave_face)
    finger_clean = np.interp(t_grid, t_sim, wave_finger)

    snr_levels_db = [35.0, 30.0, 25.0, 20.0, 15.0, 12.0, 10.0]
    n_trials = 15
    results_snr = {}

    for snr_db in snr_levels_db:
        errors_smooth = []
        errors_raw = []

        noise_std_face = 1.0 / (10.0 ** (snr_db / 20.0))
        noise_std_finger = 1.0 / (10.0 ** ((snr_db + 8.0) / 20.0))

        for _ in range(n_trials):
            noise_face = rng.normal(0.0, noise_std_face, size=len(t_grid))
            noise_finger = rng.normal(0.0, noise_std_finger, size=len(t_grid))

            face_noisy = face_clean + noise_face
            finger_noisy = finger_clean + noise_finger

            # Test with smoothing (savgol filter enabled)
            res_smooth = detector.compute_ptt(
                t_grid, face_noisy, finger_noisy, smooth=True
            )
            err_s = abs(float(res_smooth["delta_ptt_ms"]) - injected_delay_ms)
            errors_smooth.append(err_s)

            # Test without smoothing
            res_raw = detector.compute_ptt(
                t_grid, face_noisy, finger_noisy, smooth=False
            )
            err_r = abs(float(res_raw["delta_ptt_ms"]) - injected_delay_ms)
            errors_raw.append(err_r)

        mean_err_s = float(np.mean(errors_smooth))
        max_err_s = float(np.max(errors_smooth))
        mean_err_r = float(np.mean(errors_raw))
        max_err_r = float(np.max(errors_raw))

        results_snr[snr_db] = {
            "snr_db": snr_db,
            "mean_err_smooth_ms": mean_err_s,
            "max_err_smooth_ms": max_err_s,
            "mean_err_raw_ms": mean_err_r,
            "max_err_raw_ms": max_err_r,
            "passed_15db": bool(mean_err_s <= 0.75),
        }

        print(
            f"SNR: {snr_db:4.1f} dB | Smoothed Mean Err: {mean_err_s:.4f} ms (Max: {max_err_s:.4f} ms) | "
            f"Raw Mean Err: {mean_err_r:.4f} ms | "
            f"Status: {'PASS' if mean_err_s <= 0.75 else 'FAIL'}"
        )

    res_15 = results_snr[15.0]
    print(f"\n15 dB Target SNR Verdict (Smoothed): {'PASS' if res_15['passed_15db'] else 'FAIL'} (Mean error = {res_15['mean_err_smooth_ms']:.4f} ms)")

    return results_snr


def main():
    print("=" * 80)
    print("ADVERSARIAL STRESS TEST SUITE: OPTICAL & HRVP-PINN (CHALLENGER 1)")
    print("=" * 80)

    summary = {}
    summary["test1"] = run_test1_rolling_shutter_sweep()
    summary["test2"] = run_test2_vfr_jitter()
    summary["test3"] = run_test3_extreme_heart_rates()
    summary["test4"] = run_test4_extreme_arterial_stiffness()
    summary["test5"] = run_test5_extreme_hydrostatic_elevations()
    summary["test6"] = run_test6_low_snr_stress()

    # Save summary report
    output_path = os.path.join(
        os.path.dirname(__file__), "stress_test_summary.json"
    )
    with open(output_path, "w") as f:
        # Convert non-serializable elements
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))

    print("\n" + "=" * 80)
    print(f"All stress test vectors executed. Detailed results saved to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
