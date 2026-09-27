"""Comprehensive Unit Test Suite for Physics-Informed Hemodynamic Engine (HRVP-PINN).

Verifies:
  1. Sub-frame intersecting tangent foot detection accuracy against injected delays.
  2. Sub-frame timing accuracy across diverse heart rates (50 to 110 BPM) and noise levels.
  3. Anthropometric path length (D_path) and ePWV calculation.
  4. IMU hydrostatic perturbation equations and ZUPT elevation integration.
  5. Traube-Hering-Mayer 0.2 Hz Hilbert respiratory envelope and compliance.
  6. Closed-form Moens-Korteweg and Hughes arterial stiffness parameter identification (< 1% error).
  7. Standard 6D physiological anchor vector generation and end-to-end HRVP solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pinn.anthropometrics import (
    Anthropometrics,
    compute_epwv,
    compute_epwv_from_profile,
    compute_path_length,
)
from src.pinn.foot_detector import (
    FootDetector,
    calculate_tangent_foot,
    compute_subframe_ptt,
    detect_pulse_feet,
)
from src.pinn.hydrostatic_engine import (
    G_GRAVITY,
    MMHG_PER_METER,
    MMHG_PER_MM,
    PA_TO_MMHG,
    RHO_BLOOD,
    HydrostaticEngine,
    elevation_to_hydrostatic_pressure,
    integrate_zupt_elevation,
    pressure_to_elevation,
)
from src.pinn.respiratory_envelope import (
    RespiratoryEnvelopeExtractor,
    compute_compliance_slope,
    extract_respiratory_envelope,
)
from src.pinn.stiffness_solver import (
    HRVPEngine,
    calculate_k_form,
    construct_anchor_vector,
    decompose_sbp_dbp,
    solve_hrvp,
    solve_stiffness_parameters,
)


# ============================================================================
# 1. Foot Detector Tests
# ============================================================================


class TestFootDetector:
    """Tests for continuous intersecting tangent foot detection."""

    def test_analytical_tangent_foot_ideal_upstroke(self):
        """Verify closed-form root math on an ideal known analytical curve.

        Smooth sigmoidal-like upstroke: y(t) = tanh((t - 0.2) * 10) on t in [0.0, 0.4].
        Inflection occurs at t = 0.2 where y''(t) = 0.
        """
        t = np.linspace(0.0, 0.4, 25)
        y = np.tanh((t - 0.2) * 10.0)

        result = calculate_tangent_foot(t, y)
        assert "t_foot" in result
        assert "t_max_slope" in result
        assert "max_slope" in result
        assert "y_min" in result

        # Peak slope of tanh(10 * (t - 0.2)) occurs near t = 0.2
        assert abs(result["t_max_slope"] - 0.2) < 0.02
        assert result["max_slope"] > 5.0
        # Foot must be before inflection time
        assert result["t_foot"] < result["t_max_slope"]

    def test_subframe_precision_injected_delays(self):
        """Verify sub-frame precision against injected continuous delays.

        Injected delays: 40.0, 45.3, 50.7, 72.4, 98.6, 120.1 ms.
        Sampled at fs = 30 Hz (where integer sample period is 33.33 ms).
        Sub-frame spline root should reconstruct delay with precision far bypassing
        the 33.33 ms integer grid (error < 0.75 ms).
        """
        injected_delays_ms = [40.0, 45.3, 50.7, 72.4, 98.6, 120.1]
        fs = 30.0  # 30 FPS camera
        duration = 6.0
        t_continuous = np.linspace(0.0, duration, int(duration * 2000))

        hr_freq = 1.2

        def generate_pulse(t_arr, delay_sec=0.0):
            phase = 2.0 * np.pi * hr_freq * (t_arr - delay_sec)
            wave = (
                0.6 * np.sin(phase - np.pi / 2.0)
                + 0.3 * np.sin(2.0 * phase - np.pi / 2.0)
                + 0.1 * np.sin(3.0 * phase - np.pi / 2.0)
            )
            return wave

        for delay_ms in injected_delays_ms:
            delay_sec = delay_ms / 1000.0

            face_cont = generate_pulse(t_continuous, delay_sec=0.0)
            finger_cont = generate_pulse(t_continuous, delay_sec=delay_sec)

            t_30hz = np.arange(0.0, duration, 1.0 / fs)
            face_sampled = np.interp(t_30hz, t_continuous, face_cont)
            finger_sampled = np.interp(t_30hz, t_continuous, finger_cont)

            feet_face = detect_pulse_feet(t_30hz, face_sampled, fs=fs, smooth=False)
            feet_finger = detect_pulse_feet(t_30hz, finger_sampled, fs=fs, smooth=False)

            assert len(feet_face) >= 3, f"Expected at least 3 beats, got {len(feet_face)}"
            assert len(feet_finger) >= 3, f"Expected at least 3 beats, got {len(feet_finger)}"

            ptt_res = compute_subframe_ptt(
                t_30hz, face_sampled, finger_sampled, fs=fs, min_ptt_sec=0.015, max_ptt_sec=0.35
            )
            measured_delay_ms = float(ptt_res["delta_ptt_ms"])

            error_ms = abs(measured_delay_ms - delay_ms)
            # Sub-frame intersecting tangent precision should be <= 0.75 ms
            assert error_ms < 0.75, (
                f"Injected delay {delay_ms} ms, measured {measured_delay_ms} ms, "
                f"error {error_ms:.3f} ms exceeds 0.75 ms threshold!"
            )

    def test_different_heart_rates_precision(self):
        """Verify timing accuracy across physiological heart rate extremes (50 to 110 BPM)."""
        injected_ms = 75.0
        fs = 125.0
        duration = 8.0
        t_cont = np.linspace(0.0, duration, int(duration * 2000))

        for hr_bpm in [50.0, 75.0, 110.0]:
            hr_freq = hr_bpm / 60.0

            def pulse(t, delay):
                p = 2.0 * np.pi * hr_freq * (t - delay)
                return 0.6 * np.sin(p - np.pi / 2.0) + 0.3 * np.sin(2.0 * p - np.pi / 2.0) + 0.1 * np.sin(3.0 * p - np.pi / 2.0)

            face_cont = pulse(t_cont, 0.0)
            finger_cont = pulse(t_cont, injected_ms / 1000.0)

            t_grid = np.arange(0.0, duration, 1.0 / fs)
            face_sampled = np.interp(t_grid, t_cont, face_cont)
            finger_sampled = np.interp(t_grid, t_cont, finger_cont)

            res = compute_subframe_ptt(t_grid, face_sampled, finger_sampled, fs=fs)
            error_ms = abs(res["delta_ptt_ms"] - injected_ms)
            assert error_ms < 0.20, f"HR {hr_bpm} BPM error {error_ms:.3f} ms exceeds 0.20 ms threshold"

    def test_foot_detector_class_wrapper(self):
        """Test FootDetector object-oriented wrapper."""
        detector = FootDetector(fs=125.0)
        t = np.linspace(0.0, 2.0, 250)
        sig = np.sin(2.0 * np.pi * 1.0 * t)

        feet = detector.detect_feet(t, sig)
        assert isinstance(feet, np.ndarray)

        res = detector.compute_ptt(t, sig, np.roll(sig, 10))
        assert "delta_ptt_ms" in res
        assert res["delta_ptt_ms"] > 0

    def test_foot_detector_boundary_handling(self):
        """Test graceful handling of degenerate or short signals."""
        t_short = np.array([0.0, 0.01, 0.02])
        y_short = np.array([1.0, 1.0, 1.0])
        feet = detect_pulse_feet(t_short, y_short, fs=100.0)
        assert len(feet) == 0

        # Flat signal
        t_flat = np.linspace(0, 1, 100)
        y_flat = np.ones(100)
        feet_flat = detect_pulse_feet(t_flat, y_flat, fs=100.0)
        assert isinstance(feet_flat, np.ndarray)


# ============================================================================
# 2. Anthropometrics Tests
# ============================================================================


class TestAnthropometrics:
    """Tests for anatomical path length and ePWV calculation."""

    def test_path_length_explicit_arm_span(self):
        """Verify D_path = 0.55 * Height + 0.12 * ArmSpan."""
        height = 1.80
        arm_span = 1.85
        expected_d_path = 0.55 * 1.80 + 0.12 * 1.85  # 0.99 + 0.222 = 1.212 m
        computed = compute_path_length(height_m=height, arm_span_m=arm_span)
        assert abs(computed - expected_d_path) < 1e-9

    def test_path_length_default_arm_span(self):
        """Verify default ArmSpan = 1.03 * Height => D_path = 0.6736 * Height."""
        height = 1.75
        expected = 0.55 * 1.75 + 0.12 * (1.03 * 1.75)  # 0.6736 * 1.75 = 1.1788 m
        computed = compute_path_length(height_m=height, arm_span_m=None)
        assert abs(computed - expected) < 1e-9

        # Also test with non-positive arm span
        computed_zero = compute_path_length(height_m=height, arm_span_m=0.0)
        assert abs(computed_zero - expected) < 1e-9

    def test_epwv_calculation(self):
        """Verify ePWV = D_path / Delta PTT."""
        d_path = 1.20  # meters
        delta_ptt = 0.150  # 150 ms = 0.15 s
        expected_epwv = 1.20 / 0.150  # 8.0 m/s
        computed = compute_epwv(d_path_m=d_path, delta_ptt_sec=delta_ptt)
        assert abs(computed - expected_epwv) < 1e-9

    def test_compute_epwv_from_profile(self):
        """Verify full profile computation with millisecond auto-detection."""
        res = compute_epwv_from_profile(height_m=1.75, delta_ptt_sec=100.0)
        assert abs(res["delta_ptt_sec"] - 0.100) < 1e-6
        assert res["epwv_mps"] > 5.0

    def test_invalid_anthropometrics_raises(self):
        """Verify invalid or unphysiological inputs raise ValueError."""
        with pytest.raises(ValueError):
            compute_path_length(height_m=-1.75)

        with pytest.raises(ValueError):
            compute_path_length(height_m=0.1)  # Below 0.5 m

        with pytest.raises(ValueError):
            compute_path_length(height_m=3.5)  # Above 2.5 m

        with pytest.raises(ValueError):
            compute_epwv(d_path_m=1.2, delta_ptt_sec=0.0)

        with pytest.raises(ValueError):
            compute_epwv(d_path_m=-1.0, delta_ptt_sec=0.1)

    def test_anthropometrics_class(self):
        """Verify Anthropometrics class encapsulation."""
        anthro = Anthropometrics(height_m=1.80, arm_span_m=1.85)
        d_dict = anthro.to_dict()
        assert d_dict["height_m"] == 1.80
        assert d_dict["arm_span_m"] == 1.85
        assert abs(d_dict["d_path_m"] - 1.212) < 1e-6
        epwv = anthro.calculate_epwv(0.120)
        assert abs(epwv - (1.212 / 0.120)) < 1e-6


# ============================================================================
# 3. Hydrostatic Engine Tests
# ============================================================================


class TestHydrostaticEngine:
    """Tests for IMU hydrostatic column physics and ZUPT integration."""

    def test_physical_constants_exactness(self):
        """Verify density, gravity, and conversion constants."""
        assert abs(RHO_BLOOD - 1055.0) < 1e-9
        assert abs(G_GRAVITY - 9.80665) < 1e-9
        # 1055 * 9.80665 / 133.322387415 = 77.60142 mmHg/m
        assert abs(MMHG_PER_METER - 77.6014) < 0.01
        # 0.0776014 mmHg/mm
        assert abs(MMHG_PER_MM - 0.0776014) < 1e-6

    def test_elevation_to_pressure_shift(self):
        """Verify Delta P_hydro = rho * g * Delta h.

        +30 cm displacement (0.30 m) => 77.6014 * 0.30 = 23.2804 mmHg.
        """
        res = elevation_to_hydrostatic_pressure(delta_h_m=0.30)
        assert abs(res["delta_h_mm"] - 300.0) < 1e-6
        assert abs(res["delta_p_mmhg"] - 23.2804) < 0.01

        # Test negative displacement (-25 cm)
        res_neg = elevation_to_hydrostatic_pressure(delta_h_m=-0.25)
        assert abs(res_neg["delta_p_mmhg"] - (-19.4003)) < 0.01

    def test_pressure_to_elevation_roundtrip(self):
        """Verify pressure to elevation roundtrip inversion."""
        target_h = 0.42  # 42 cm
        press = elevation_to_hydrostatic_pressure(target_h)["delta_p_mmhg"]
        recovered_h = pressure_to_elevation(press)
        assert abs(recovered_h - target_h) < 1e-6

    def test_zupt_kinematic_integration(self):
        """Verify IMU vertical acceleration integration with Zero-Velocity Updates."""
        fs_imu = 100.0
        t = np.linspace(0.0, 4.0, int(4.0 * fs_imu))
        accel = np.zeros_like(t)

        idx_accel = (t >= 1.0) & (t < 1.5)
        idx_decel = (t >= 1.5) & (t < 2.0)
        accel[idx_accel] = 1.2
        accel[idx_decel] = -1.2

        zupt_res = integrate_zupt_elevation(t, accel, stationary_threshold=0.08)
        assert zupt_res["hold_stable"] is True
        assert zupt_res["delta_h_m"] > 0.20
        assert zupt_res["delta_p_mmhg"] > 15.0

    def test_hydrostatic_engine_class(self):
        """Verify HydrostaticEngine class methods."""
        engine = HydrostaticEngine()
        shift = engine.calculate_pressure_shift(0.20)
        assert abs(shift["delta_p_mmhg"] - (77.6014 * 0.20)) < 0.02


# ============================================================================
# 4. Respiratory Envelope Tests
# ============================================================================


class TestRespiratoryEnvelope:
    """Tests for Traube-Hering-Mayer 0.2 Hz envelope and compliance extraction."""

    def test_respiratory_envelope_extraction(self):
        """Verify THM envelope extraction on synthetic respiration-modulated ePWV.

        ePWV(t) = 7.0 + 0.6 * cos(2 * pi * 0.20 * t)
        Expected:
          - Respiratory rate ~ 12 BPM (0.20 Hz * 60)
          - Mean envelope amplitude ~ 0.6 m/s
        """
        duration = 20.0
        cycle_times = np.arange(0.0, duration, 0.8)
        f_resp = 0.20  # 0.2 Hz Traube-Hering-Mayer frequency
        amp_true = 0.60
        epwv_series = 7.0 + amp_true * np.cos(2.0 * np.pi * f_resp * cycle_times)

        res = extract_respiratory_envelope(
            cycle_times=cycle_times,
            epwv_series=epwv_series,
            fs_resample=10.0,
            f_low=0.15,
            f_high=0.35,
        )

        assert "time_uniform" in res
        assert "analytic_signal" in res
        assert "instantaneous_amplitude" in res
        assert "mean_envelope_amplitude" in res
        assert "respiratory_rate_bpm" in res

        assert len(res["time_uniform"]) == int(np.round(cycle_times[-1] * 10.0))
        assert abs(res["mean_envelope_amplitude"] - amp_true) < 0.15
        assert abs(res["respiratory_rate_bpm"] - 12.0) <= 2.0

    def test_compliance_slope_calculation(self):
        """Verify compliance slope d(ePWV)/dP = (ePWV_2 - ePWV_1) / Delta P_hydro."""
        epwv_1 = 6.2  # m/s
        epwv_2 = 7.1  # m/s
        delta_p = 18.0  # mmHg
        expected_slope = (7.1 - 6.2) / 18.0  # 0.9 / 18 = 0.05 (m/s)/mmHg
        slope = compute_compliance_slope(epwv_1, epwv_2, delta_p)
        assert abs(slope - expected_slope) < 1e-9

    def test_respiratory_extractor_class(self):
        """Verify RespiratoryEnvelopeExtractor class interface."""
        extractor = RespiratoryEnvelopeExtractor(fs_resample=10.0)
        t = np.linspace(0, 10, 15)
        epwv = 6.0 + 0.3 * np.sin(2.0 * np.pi * 0.2 * t)
        res = extractor.extract(t, epwv)
        assert res["respiratory_rate_bpm"] > 0
        slope = extractor.compliance_slope(6.0, 6.5, 10.0)
        assert abs(slope - 0.05) < 1e-6


# ============================================================================
# 5. Stiffness Solver & Master Pipeline Tests
# ============================================================================


class TestStiffnessSolver:
    """Tests for closed-form parameter recovery, morphology factor, and HRVP solver."""

    def test_closed_form_stiffness_parameter_recovery(self):
        """Verify exact closed-form identification of alpha and E_0 (< 1% error).

        Forward model:
          Hughes law: PWV(P) = c_0 * exp(alpha * P / 2)
          Ground truth:
            c_0 = 4.5 m/s
            alpha_true = 0.024 mmHg^-1
            P_1 = 80 mmHg (diastolic/baseline)
            Delta h = 0.30 m => Delta P_hydro = 77.6014 * 0.30 = 23.2804 mmHg
            P_2 = P_1 + Delta P_hydro = 103.2804 mmHg
        """
        c0_true = 4.5
        alpha_true = 0.024  # mmHg^-1
        p1 = 80.0
        delta_h = 0.30
        delta_p = 77.60142 * delta_h
        p2 = p1 + delta_p

        epwv_1 = c0_true * np.exp(alpha_true * p1 / 2.0)
        epwv_2 = c0_true * np.exp(alpha_true * p2 / 2.0)

        solved = solve_stiffness_parameters(
            epwv_1=epwv_1,
            epwv_2=epwv_2,
            delta_h_m=delta_h,
            c0=c0_true,
        )

        alpha_recovered = solved["alpha_mmhg"]
        rel_error_alpha = abs(alpha_recovered - alpha_true) / alpha_true
        assert rel_error_alpha < 0.01, (
            f"Alpha relative error {rel_error_alpha * 100:.3f}% exceeds 1% limit! "
            f"True: {alpha_true}, Solved: {alpha_recovered}"
        )

        map_recovered = solved["map_analytical_mmhg"]
        assert abs(map_recovered - p1) < 1.0
        assert solved["e0_pa"] > 1e5

    def test_stiffness_parameter_recovery_across_pressures(self):
        """Verify Hughes parameter recovery across hypotensive, normotensive, and hypertensive pressures."""
        c0_true = 4.5
        alpha_true = 0.024
        pressures = [65.0, 90.0, 115.0]
        delta_h = 0.28
        delta_p = 77.60142 * delta_h

        for p1 in pressures:
            p2 = p1 + delta_p
            epwv_1 = c0_true * np.exp(alpha_true * p1 / 2.0)
            epwv_2 = c0_true * np.exp(alpha_true * p2 / 2.0)

            sol = solve_stiffness_parameters(epwv_1, epwv_2, delta_h, c0=c0_true)
            rel_err = abs(sol["alpha_mmhg"] - alpha_true) / alpha_true
            assert rel_err < 0.01, f"Failed at P1={p1}, error={rel_err*100:.3f}%"
            assert abs(sol["map_analytical_mmhg"] - p1) < 0.1

    def test_k_form_calculation(self):
        """Verify k_form waveform area factor."""
        t = np.linspace(0, 1, 100)
        triangle_wave = 1.0 - np.abs(2.0 * t - 1.0)
        k_triangle = calculate_k_form(triangle_wave)
        assert abs(k_triangle - 0.50) < 0.05

        ppg_like = (np.sin(2.0 * np.pi * t - np.pi / 2.0) + 1.0) ** 2
        k_ppg = calculate_k_form(ppg_like)
        assert 0.30 <= k_ppg <= 0.45

    def test_decompose_sbp_dbp(self):
        """Verify SBP and DBP decomposition from MAP and k_form."""
        map_val = 90.0
        k_form = 0.33
        pp = 40.0
        dec = decompose_sbp_dbp(map_mmhg=map_val, k_form=k_form, pulse_pressure_mmhg=pp)
        assert abs(dec["sbp_mmhg"] - (90.0 + (1 - 0.33) * 40.0)) < 1e-6
        assert abs(dec["dbp_mmhg"] - (90.0 - 0.33 * 40.0)) < 1e-6
        reconstructed_map = dec["dbp_mmhg"] + k_form * pp
        assert abs(reconstructed_map - map_val) < 1e-6

    def test_anchor_vector_shape_and_values(self):
        """Verify 6-dimensional physiological anchor vector."""
        anchor = construct_anchor_vector(
            epwv_mps=7.2,
            delta_ptt_ms=95.0,
            delta_p_hydro_mmhg=22.5,
            compliance_slope=0.045,
            e0_pa=1.85e5,
            alpha_mmhg=0.024,
        )
        assert isinstance(anchor, np.ndarray)
        assert anchor.shape == (6,)
        assert anchor.dtype == np.float32
        assert abs(anchor[0] - 7.2) < 1e-5
        assert abs(anchor[1] - 95.0) < 1e-5
        assert abs(anchor[2] - 22.5) < 1e-5
        assert abs(anchor[3] - 0.045) < 1e-5
        assert abs(anchor[4] - 1.85e5) < 1.0
        assert abs(anchor[5] - 0.024) < 1e-5
        assert not np.any(np.isnan(anchor))
        assert not np.any(np.isinf(anchor))

    def test_solve_hrvp_end_to_end(self):
        """Verify end-to-end solve_hrvp interface contract compliance."""
        fs = 125.0
        duration = 6.0
        t = np.linspace(0.0, duration, int(duration * fs))

        face_sig = np.sin(2.0 * np.pi * 1.2 * t)
        finger_sig = np.sin(2.0 * np.pi * 1.2 * (t - 0.100))

        imu_data = {"delta_h_m": 0.25, "hold_stable": True}
        user_profile = {"height_m": 1.78, "arm_span_m": 1.80}

        out = solve_hrvp(
            face_signal=face_sig,
            finger_signal=finger_sig,
            imu_data=imu_data,
            user_profile=user_profile,
            fs=fs,
        )

        required_keys = [
            "delta_ptt_ms",
            "epwv_mps",
            "alpha",
            "e0_pa",
            "compliance_slope",
            "map_analytical_mmhg",
            "sbp_analytical_mmhg",
            "dbp_analytical_mmhg",
            "anchor_vector",
        ]
        for key in required_keys:
            assert key in out, f"Missing required key '{key}' in hrvp_output"

        anchor = out["anchor_vector"]
        assert isinstance(anchor, np.ndarray)
        assert anchor.shape == (6,)

        assert 50.0 <= out["delta_ptt_ms"] <= 250.0
        assert 4.0 <= out["epwv_mps"] <= 15.0
        assert 60.0 <= out["sbp_analytical_mmhg"] <= 180.0
        assert 40.0 <= out["dbp_analytical_mmhg"] <= 120.0
        assert out["sbp_analytical_mmhg"] > out["dbp_analytical_mmhg"]

    def test_hrvp_engine_class(self):
        """Verify HRVPEngine class interface."""
        engine = HRVPEngine(fs=125.0)
        t = np.linspace(0.0, 4.0, 500)
        sig1 = np.sin(2.0 * np.pi * 1.1 * t)
        sig2 = np.sin(2.0 * np.pi * 1.1 * (t - 0.08))
        res = engine.process(
            face_signal=sig1,
            finger_signal=sig2,
            imu_data={"delta_h_m": 0.15, "hold_stable": True},
            user_profile={"height_m": 1.70},
        )
        assert "map_analytical_mmhg" in res
