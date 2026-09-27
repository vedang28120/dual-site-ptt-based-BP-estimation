"""
Detailed investigation of Vector 1: Rolling-shutter phase offset and PCHIP resampling.
"""

import sys
import numpy as np

PROJECT_ROOT = r"C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.optical.time_grid_synchronizer import TimeGridSynchronizer
from src.pinn.foot_detector import FootDetector, calculate_tangent_foot
from tests.synthetic_generator import generate_single_cardiac_cycle

def test_isolated_phase():
    detector = FootDetector(fs=125.0)
    synchronizer = TimeGridSynchronizer(target_fs_hz=125.0)

    true_delay_ms = 60.0
    true_delay_sec = true_delay_ms / 1000.0
    hr_bpm = 75.0
    duration = 10.0
    fps = 30.0
    nominal_dt = 1.0 / fps

    t_sim = np.linspace(0.0, duration, int(duration * 10000))
    t_cardiac = 60.0 / hr_bpm
    face_phase = (t_sim % t_cardiac) / t_cardiac
    finger_phase = ((t_sim - true_delay_sec) % t_cardiac) / t_cardiac

    # Case A: Ideal smooth sinusoidal multi-harmonic wave (like in test_pinn.py)
    p_f = 2.0 * np.pi * (hr_bpm / 60.0) * t_sim
    p_d = 2.0 * np.pi * (hr_bpm / 60.0) * (t_sim - true_delay_sec)
    sine_face = 0.6 * np.sin(p_f - np.pi/2) + 0.3 * np.sin(2*p_f - np.pi/2) + 0.1 * np.sin(3*p_f - np.pi/2)
    sine_finger = 0.6 * np.sin(p_d - np.pi/2) + 0.3 * np.sin(2*p_d - np.pi/2) + 0.1 * np.sin(3*p_d - np.pi/2)

    # Case B: Multi-Gaussian physiological pulse (synthetic_generator.py)
    gauss_face = generate_single_cardiac_cycle(face_phase, is_distal=False, damping_factor=0.3)
    gauss_finger = generate_single_cardiac_cycle(finger_phase, is_distal=True)

    print("--- Testing Phase Sweep on Sine Wave vs Multi-Gaussian ---")
    offsets = [0.0, 5.0, 10.0, 16.666, 20.0, 25.0, 33.333]

    for phi_ms in offsets:
        phi_sec = phi_ms / 1000.0
        t_front = np.arange(0.0, duration, nominal_dt)
        t_rear = np.arange(phi_sec, duration, nominal_dt)

        t_front_ns = (t_front * 1e9).astype(np.int64)
        t_rear_ns = (t_rear * 1e9).astype(np.int64)

        # Sine wave
        s_face_samp = np.interp(t_front, t_sim, sine_face)
        s_finger_samp = np.interp(t_rear, t_sim, sine_finger)

        f_rgb = np.zeros((len(t_front), 3), dtype=np.float32)
        f_rgb[:, 1] = s_face_samp
        r_rgb = np.zeros((len(t_rear), 3), dtype=np.float32)
        r_rgb[:, 1] = s_finger_samp

        sync_sine = synchronizer.synchronize_streams(t_front_ns, f_rgb, t_rear_ns, r_rgb)
        ptt_sine = detector.compute_ptt(
            sync_sine.master_time_sec, sync_sine.resampled_face[:, 1], sync_sine.resampled_finger[:, 1]
        )
        err_sine = abs(float(ptt_sine["delta_ptt_ms"]) - true_delay_ms)

        # Gaussian wave
        g_face_samp = np.interp(t_front, t_sim, gauss_face)
        g_finger_samp = np.interp(t_rear, t_sim, gauss_finger)
        f_rgb[:, 1] = g_face_samp
        r_rgb[:, 1] = g_finger_samp

        sync_gauss = synchronizer.synchronize_streams(t_front_ns, f_rgb, t_rear_ns, r_rgb)
        ptt_gauss = detector.compute_ptt(
            sync_gauss.master_time_sec, sync_gauss.resampled_face[:, 1], sync_gauss.resampled_finger[:, 1]
        )
        err_gauss = abs(float(ptt_gauss["delta_ptt_ms"]) - true_delay_ms)

        print(f"Offset: {phi_ms:6.2f} ms | Sine Err: {err_sine:.4f} ms | Gaussian Err: {err_gauss:.4f} ms")

if __name__ == "__main__":
    test_isolated_phase()
