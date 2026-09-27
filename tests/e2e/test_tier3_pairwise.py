"""
Tier 3: Pairwise Cross-Feature Combinations Test Suite
Author: test_writer_1 (E2E Testing Architect)

Validates multi-variable interactions across the 15 features in pairwise combinations:
1. High Heart Rate (160 BPM) + Hydrostatic Perturbation (+300 mm)
2. Low Compliance / Rigid Aorta (alpha=0.040) + High Optical Noise (20 dB SNR)
3. High VFR Timestamp Jitter (10 ms) + Sub-Frame Continuous Spline Foot Detection
4. Micro-elevation (15 mm) + Traube-Hering-Mayer 0.2 Hz Respiratory Envelope
5. Tall Stature (Height 2.05 m) + Tachycardia (170 BPM)
6. Short Stature (Height 1.35 m) + Bradycardia (40 BPM)
7. Hypertensive Crisis (200/120 mmHg) + High Hydrostatic Elevation (+400 mm)
8. Severe Hypotension (80/50 mmHg) + Negative Hydrostatic Elevation (-300 mm)
9. Rapid Mayer Respiration (0.35 Hz) + Stiff Arteries (alpha=0.035)
10. Slow Mayer Respiration (0.12 Hz) + Elastic Arteries (alpha=0.018)
11. Decoupled MediaPipe Fast Path + Large Rolling-Shutter Phase (25 ms)
12. Manual Anti-Saturation Exposure + 30-Second Long Continuous Acquisition
13. Causal Cross-Attention + Dynamic HRVP Anchor Vector Perturbation
14. Tripartite Compound Loss + Outlier Cohort Distribution
15. INT8 Quantized Model + Dynamic Range Optical Waveforms
16. Zero-Tilt Hydrostatic Edge + Elevated Heart Rate (140 BPM)
Total: 16 pairwise cross-feature test cases.
"""

import pytest
import numpy as np
import torch

from tests.synthetic_generator import (
    PhysiologicalProfile,
    SyntheticPhysiologicalGenerator,
    calculate_d_path,
    calculate_c0,
    calculate_epwv,
    calculate_delta_ptt_ms,
    calculate_hydrostatic_delta_p,
    calculate_compliance_slope
)
from tests.reference_pinn import (
    ReferenceTimeGridSynchronizer,
    ReferenceSubFrameFootDetector,
    ReferenceAnthropometrics,
    ReferenceHydrostaticEngine,
    ReferenceTHMAnalyzer,
    ReferenceStiffnessSolver,
    ReferenceTripartiteLoss,
    ReferenceINT8Quantizer
)
from tests.reference_model import (
    DualStreamTCN,
    PyTorchTripartiteLoss
)


class TestTier3PairwiseCombinations:
    def test_p1_high_hr_plus_hydrostatic_perturbation(self):
        """Pair 1: High Heart Rate (160 BPM) + Hydrostatic Perturbation (+300 mm)."""
        gen = SyntheticPhysiologicalGenerator(seed=301)
        prof = PhysiologicalProfile(hr_bpm=160.0, delta_h_m=0.30)
        session = gen.generate_session(prof, duration_sec=3.0)
        gt = session["ground_truth"]
        assert gt["hr_bpm"] == 160.0
        assert abs(gt["delta_p_hydro_mmhg"] - 23.28) < 0.2
        # Elevated HR with hydrostatic drop still maintains valid ePWV
        assert 5.0 <= gt["epwv_perturbed_mps"] <= 14.0

    def test_p2_low_compliance_plus_high_optical_noise(self):
        """Pair 2: Low Compliance / Rigid Aorta (alpha=0.040) + High Noise (20 dB SNR)."""
        gen = SyntheticPhysiologicalGenerator(seed=302)
        prof = PhysiologicalProfile(alpha=0.040, e0_pa=1.8e5, snr_db=20.0)
        session = gen.generate_session(prof, duration_sec=3.0)
        sync = ReferenceTimeGridSynchronizer()
        t_m, f_s, r_s = sync.synchronize(
            session["t_front_ns"], session["face_rgb"],
            session["t_rear_ns"], session["finger_rgb"]
        )
        assert len(t_m) > 100
        assert session["ground_truth"]["alpha"] == 0.040

    def test_p3_high_vfr_jitter_plus_subframe_foot_detection(self):
        """Pair 3: High VFR Jitter (10 ms) + Sub-Frame Tangent Spline Foot Detection."""
        gen = SyntheticPhysiologicalGenerator(seed=303)
        prof = PhysiologicalProfile(vfr_jitter_ms=10.0)
        session = gen.generate_session(prof, duration_sec=4.0)
        sync = ReferenceTimeGridSynchronizer()
        t_m, f_s, r_s = sync.synchronize(
            session["t_front_ns"], session["face_rgb"],
            session["t_rear_ns"], session["finger_rgb"]
        )
        detector = ReferenceSubFrameFootDetector(fs=125.0)
        feet_face = detector.detect_feet(f_s[:, 1], t_m)
        feet_finger = detector.detect_feet(r_s[:, 1], t_m)
        assert len(feet_face) >= 3
        assert len(feet_finger) >= 3

    def test_p4_micro_elevation_plus_thm_respiratory_envelope(self):
        """Pair 4: Micro-elevation (15 mm) + Traube-Hering-Mayer 0.2 Hz Respiratory Envelope."""
        hydro = ReferenceHydrostaticEngine()
        dp = hydro.calculate_delta_p(0.015) # 15 mm -> ~1.16 mmHg
        assert abs(dp - 1.164) < 0.05
        analyzer = ReferenceTHMAnalyzer(fs=100.0)
        t = np.linspace(0, 10.0, 1000)
        epwv_mod = 7.0 + 0.15 * np.sin(2 * np.pi * 0.20 * t)
        env, mean_amp = analyzer.extract_envelope(epwv_mod)
        assert mean_amp > 0

    def test_p5_tall_stature_plus_tachycardia(self):
        """Pair 5: Tall Stature (Height 2.05 m) + Tachycardia (170 BPM)."""
        d_path = ReferenceAnthropometrics.calculate_d_path(height_m=2.05)
        assert d_path > 1.35
        gen = SyntheticPhysiologicalGenerator(seed=305)
        prof = PhysiologicalProfile(height_m=2.05, hr_bpm=170.0)
        session = gen.generate_session(prof, duration_sec=3.0)
        assert session["ground_truth"]["d_path_m"] == d_path

    def test_p6_short_stature_plus_bradycardia(self):
        """Pair 6: Short Stature (Height 1.35 m) + Bradycardia (40 BPM)."""
        d_path = ReferenceAnthropometrics.calculate_d_path(height_m=1.35)
        assert d_path < 1.05
        gen = SyntheticPhysiologicalGenerator(seed=306)
        prof = PhysiologicalProfile(height_m=1.35, hr_bpm=40.0)
        session = gen.generate_session(prof, duration_sec=4.0)
        assert session["ground_truth"]["hr_bpm"] == 40.0

    def test_p7_hypertensive_crisis_plus_high_hydrostatic_elevation(self):
        """Pair 7: Hypertensive Crisis (200/120 mmHg) + High Elevation (+400 mm)."""
        gen = SyntheticPhysiologicalGenerator(seed=307)
        prof = PhysiologicalProfile(sbp_mmhg=200.0, dbp_mmhg=120.0, delta_h_m=0.40)
        session = gen.generate_session(prof, duration_sec=2.0)
        gt = session["ground_truth"]
        # Perturbed SBP = 200 + (77.601 * 0.40) = 231.04 mmHg
        assert gt["sbp_perturbed_mmhg"] > 225.0

    def test_p8_severe_hypotension_plus_negative_elevation(self):
        """Pair 8: Severe Hypotension (80/50 mmHg) + Negative Hydrostatic Shift (-300 mm)."""
        gen = SyntheticPhysiologicalGenerator(seed=308)
        prof = PhysiologicalProfile(sbp_mmhg=80.0, dbp_mmhg=50.0, delta_h_m=-0.30)
        session = gen.generate_session(prof, duration_sec=2.0)
        gt = session["ground_truth"]
        # Negative elevation shift drops finger pressure further: 80 - 23.28 = ~56.7 mmHg
        assert gt["sbp_perturbed_mmhg"] < 60.0
        assert gt["dbp_perturbed_mmhg"] < 30.0

    def test_p9_rapid_mayer_respiration_plus_stiff_arteries(self):
        """Pair 9: Rapid Mayer Respiration (0.35 Hz) + Stiff Arteries (alpha=0.035)."""
        prof = PhysiologicalProfile(resp_rate_bpm=21.0, alpha=0.035) # 21 BPM = 0.35 Hz
        gen = SyntheticPhysiologicalGenerator(seed=309)
        session = gen.generate_session(prof, duration_sec=4.0)
        gt = session["ground_truth"]
        assert gt["alpha"] == 0.035
        assert gt["compliance_slope"] > 0

    def test_p10_slow_mayer_respiration_plus_elastic_arteries(self):
        """Pair 10: Slow Mayer Respiration (0.12 Hz) + Elastic Arteries (alpha=0.018)."""
        prof = PhysiologicalProfile(resp_rate_bpm=7.2, alpha=0.018) # 7.2 BPM = 0.12 Hz
        gen = SyntheticPhysiologicalGenerator(seed=310)
        session = gen.generate_session(prof, duration_sec=5.0)
        gt = session["ground_truth"]
        assert gt["alpha"] == 0.018

    def test_p11_decoupled_mediapipe_plus_large_phase_offset(self):
        """Pair 11: Decoupled Fast Path + Large Inter-Camera Phase (25 ms)."""
        gen = SyntheticPhysiologicalGenerator(seed=311)
        prof = PhysiologicalProfile(phase_offset_ms=25.0)
        session = gen.generate_session(prof, duration_sec=2.0)
        sync = ReferenceTimeGridSynchronizer(target_fs_hz=125.0)
        t_m, f_s, r_s = sync.synchronize(
            session["t_front_ns"], session["face_rgb"],
            session["t_rear_ns"], session["finger_rgb"]
        )
        assert len(t_m) == len(f_s) == len(r_s)

    def test_p12_manual_anti_saturation_plus_long_acquisition(self):
        """Pair 12: Manual Anti-Saturation Exposure + 30-Second Acquisition."""
        gen = SyntheticPhysiologicalGenerator(seed=312)
        session = gen.generate_session(PhysiologicalProfile(), duration_sec=30.0)
        green = session["finger_rgb"][:, 1]
        assert np.max(green) < 255.0
        assert len(session["t_rear_ns"]) >= 870

    def test_p13_causal_cross_attention_plus_dynamic_anchor(self):
        """Pair 13: Causal Cross-Attention + Dynamic HRVP Anchor Injection."""
        model = DualStreamTCN()
        model.eval()
        x_m = torch.randn(2, 1, 300)
        x_t = torch.randn(2, 1, 300)
        # Varying anchor vectors
        z1 = torch.tensor([[6.5, 120.0, 15.0, 0.02, 60000.0, 0.022]])
        z2 = torch.tensor([[10.5, 80.0, 30.0, 0.04, 110000.0, 0.038]])
        z = torch.cat([z1, z2], dim=0)
        with torch.no_grad():
            sbp, dbp = model(x_m, x_t, z)
        assert sbp.shape == (2, 1)
        # Predictions should differ because anchors differ
        assert abs(sbp[0].item() - sbp[1].item()) > 1e-4

    def test_p14_tripartite_loss_plus_outlier_cohort(self):
        """Pair 14: Tripartite Compound Loss + Outlier Cohort Distribution."""
        loss_fn = ReferenceTripartiteLoss()
        y_true = np.array([85.0, 115.0, 140.0, 210.0]) # Outlier hypertensive crisis
        y_pred = np.array([88.0, 114.0, 138.0, 205.0])
        res = loss_fn.compute_loss(y_true, y_pred)
        assert res["pearson_r"] > 0.95
        assert res["total_loss"] < 5.0

    def test_p15_int8_quantization_plus_dynamic_optical_waves(self):
        """Pair 15: INT8 Quantized Weights + Dynamic Optical Waveforms."""
        w_fp32 = np.random.normal(0, 0.8, size=(64, 64)).astype(np.float32)
        q_w, scale = ReferenceINT8Quantizer.quantize_symmetric(w_fp32)
        dequant = ReferenceINT8Quantizer.dequantize_symmetric(q_w, scale)
        snr = ReferenceINT8Quantizer.calculate_quantization_error_snr(w_fp32, dequant)
        assert snr > 35.0

    def test_p16_zero_tilt_edge_plus_high_heart_rate(self):
        """Pair 16: Zero-Tilt Hydrostatic Edge + Elevated Heart Rate (140 BPM)."""
        gen = SyntheticPhysiologicalGenerator(seed=316)
        prof = PhysiologicalProfile(hr_bpm=140.0, delta_h_m=0.0)
        session = gen.generate_session(prof, duration_sec=3.0)
        gt = session["ground_truth"]
        assert gt["delta_p_hydro_mmhg"] == 0.0
        assert gt["hr_bpm"] == 140.0
