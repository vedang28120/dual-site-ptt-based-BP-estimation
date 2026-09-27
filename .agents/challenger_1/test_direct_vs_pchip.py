"""
Test hypothesis: Direct continuous cubic spline on raw timestamps vs PCHIP 125 Hz resampling.
"""

import sys
import numpy as np

PROJECT_ROOT = r"C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.optical.time_grid_synchronizer import TimeGridSynchronizer
from src.pinn.foot_detector import FootDetector, detect_pulse_feet, compute_subframe_ptt
from tests.synthetic_generator import generate_single_cardiac_cycle

def test_direct_vs_pchip():
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

    wave_face = generate_single_cardiac_cycle(face_phase, is_distal=False, damping_factor=0.3)
    wave_finger = generate_single_cardiac_cycle(finger_phase, is_distal=True)

    offsets = [0.0, 5.0, 10.0, 16.666, 20.0, 25.0, 33.333]

    print("--- Comparing PCHIP (125 Hz) vs Direct Spline on Raw 30 FPS ---")
    for phi_ms in offsets:
        phi_sec = phi_ms / 1000.0
        t_front = np.arange(0.0, duration, nominal_dt)
        t_rear = np.arange(phi_sec, duration, nominal_dt)

        t_front_ns = (t_front * 1e9).astype(np.int64)
        t_rear_ns = (t_rear * 1e9).astype(np.int64)

        face_samp = np.interp(t_front, t_sim, wave_face)
        finger_samp = np.interp(t_rear, t_sim, wave_finger)

        # 1. PCHIP Resampling (Current Architecture)
        f_rgb = np.zeros((len(t_front), 3), dtype=np.float32)
        f_rgb[:, 1] = face_samp
        r_rgb = np.zeros((len(t_rear), 3), dtype=np.float32)
        r_rgb[:, 1] = finger_samp

        sync_res = synchronizer.synchronize_streams(t_front_ns, f_rgb, t_rear_ns, r_rgb)
        ptt_pchip = compute_subframe_ptt(
            sync_res.master_time_sec, sync_res.resampled_face[:, 1], sync_res.resampled_finger[:, 1]
        )
        err_pchip = abs(float(ptt_pchip["delta_ptt_ms"]) - true_delay_ms)

        # 2. Direct Spline on Raw Timestamps
        feet_face_raw = detect_pulse_feet(t_front, face_samp, fs=30.0, smooth=False)
        feet_finger_raw = detect_pulse_feet(t_rear, finger_samp, fs=30.0, smooth=False)

        # Match beats
        delays = []
        for tf in feet_face_raw:
            cands = feet_finger_raw[(feet_finger_raw >= tf + 0.015) & (feet_finger_raw <= tf + 0.35)]
            if len(cands) > 0:
                delays.append(cands[0] - tf)
        if delays:
            err_direct = abs(np.median(delays) * 1000.0 - true_delay_ms)
        else:
            err_direct = 999.0

        print(f"Offset {phi_ms:5.2f} ms | PCHIP Err: {err_pchip:6.3f} ms | Direct Spline Err: {err_direct:6.3f} ms")

if __name__ == "__main__":
    test_direct_vs_pchip()
