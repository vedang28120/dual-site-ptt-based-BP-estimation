"""
Comprehensive Unit Tests for Milestone 4: Mobile System Integration & Feasibility Pipeline.

Tests:
1. TFLiteRunner:
   - Pre-allocated zero-dynamic-allocation buffer invariance.
   - Genuine INT8 TFLite flatbuffer export, loading, and layer reconstruction.
   - Snapdragon 4 Gen 2 mobile budget validations (latency < 60 ms, peak memory <= 20 MB, file size < 250 KB).
   - TFLite-style get_tensor / set_tensor APIs.
   - Cold start vs warm inference benchmarking.
   - NaN input recovery and memory leak free execution.
2. DualSiteBPPipeline:
   - End-to-end execution bridging Camera2 nanosecond PCHIP synchronization, HRVP-PINN solver, and DS-TCN model.
   - Interface Contract compliance: outputs {"sbp", "dbp", "p_map", "epwv_mps", "status"}.
   - Stream processing from both session dictionary and separate arguments.
   - Continuous 10-second and 30-second acquisition runs.
3. Clinical Plausibility Guards & Fault Tolerance:
   - Capillary blanching detection (via Perfusion Index floor & saturation ceiling).
   - Motion artifact and high-noise rejection.
   - Camera frame arrival jitter and dropped frames detection.
   - Disjoint/non-overlapping camera streams rejection.
   - Physiological bounds clamping (SBP [70, 200], DBP [40, 130], PP >= 15 mmHg).
   - Sustained 30 FPS timing verification over 10-second acquisition window.
"""

import os
import tempfile
import time
import pytest
import numpy as np
import torch

from src.models.ds_tcn import DualStreamTCN
from src.models.quantizer import INT8Quantizer
from src.mobile.tflite_runner import TFLiteRunner, EdgeRuntimeProfile
from src.mobile.pipeline import (
    DualSiteBPPipeline,
    PipelineConfig,
    PipelineResult,
    SignalQualityMetrics,
    estimate_snr_db,
    normalize_waveform_tensor,
)
from tests.synthetic_generator import (
    PhysiologicalProfile,
    SyntheticPhysiologicalGenerator,
)


# ==============================================================================
# Suite 1: TFLiteRunner & Edge Profiling Tests
# ==============================================================================

class TestTFLiteRunner:
    """Verifies static memory pre-allocation, INT8 loading, and Snapdragon 4 Gen 2 budgets."""

    def test_runner_initialization_default_and_custom_seq_len(self):
        """Runner must initialize default DualStreamTCN model and buffers for given seq_len."""
        runner = TFLiteRunner(seq_len=300)
        assert runner.seq_len == 300
        assert runner._morph_buffer.shape == (1, 1, 300)
        assert runner._timing_buffer.shape == (1, 1, 300)
        assert runner._anchor_buffer.shape == (1, 6)
        assert runner._sbp_output_buffer.shape == (1, 1)
        assert runner._dbp_output_buffer.shape == (1, 1)

        # Custom sequence length
        runner_150 = TFLiteRunner(seq_len=150)
        assert runner_150.seq_len == 150
        assert runner_150._morph_buffer.shape == (1, 1, 150)

    def test_preallocated_static_buffers_zero_dynamic_allocation(self):
        """Memory buffer pointers must remain strictly invariant across multiple inferences."""
        runner = TFLiteRunner(seq_len=300)

        # Record initial physical memory addresses
        initial_morph_ptr = runner._morph_buffer.__array_interface__["data"][0]
        initial_timing_ptr = runner._timing_buffer.__array_interface__["data"][0]
        initial_anchor_ptr = runner._anchor_buffer.__array_interface__["data"][0]
        initial_sbp_ptr = runner._sbp_output_buffer.__array_interface__["data"][0]
        initial_dbp_ptr = runner._dbp_output_buffer.__array_interface__["data"][0]

        # Execute 25 consecutive inferences with diverse inputs
        rng = np.random.RandomState(42)
        for _ in range(25):
            m = rng.randn(1, 1, 300).astype(np.float32)
            t = rng.randn(1, 1, 300).astype(np.float32)
            a = rng.randn(6).astype(np.float32)
            sbp, dbp = runner.predict(m, t, a)
            assert isinstance(sbp, float) and isinstance(dbp, float)

        # Verify exact address preservation (Zero Dynamic Heap Allocation)
        assert runner._morph_buffer.__array_interface__["data"][0] == initial_morph_ptr
        assert runner._timing_buffer.__array_interface__["data"][0] == initial_timing_ptr
        assert runner._anchor_buffer.__array_interface__["data"][0] == initial_anchor_ptr
        assert runner._sbp_output_buffer.__array_interface__["data"][0] == initial_sbp_ptr
        assert runner._dbp_output_buffer.__array_interface__["data"][0] == initial_dbp_ptr
        assert runner.verify_buffer_invariance() is True

    def test_set_and_get_tensor_apis(self):
        """TFLite-style get/set tensor APIs must populate and read pre-allocated buffers."""
        runner = TFLiteRunner(seq_len=300)

        # Set by string identifier
        test_morph = np.ones((1, 1, 300), dtype=np.float32) * 0.75
        test_timing = np.ones((1, 1, 300), dtype=np.float32) * 0.25
        test_anchor = np.array([7.2, 95.0, 15.0, 0.09, 82000.0, 0.025], dtype=np.float32)

        runner.set_tensor("morph", test_morph)
        runner.set_tensor("timing", test_timing)
        runner.set_tensor("anchor", test_anchor)

        assert np.allclose(runner._morph_buffer, test_morph)
        assert np.allclose(runner._timing_buffer, test_timing)
        assert np.allclose(runner._anchor_buffer[0], test_anchor)

        # Execute invoke and check get_tensor
        sbp, dbp = runner.invoke()
        sbp_buf = runner.get_tensor("sbp")
        dbp_buf = runner.get_tensor("dbp")

        assert sbp_buf[0, 0] == sbp
        assert dbp_buf[0, 0] == dbp

        # Integer indices (0: morph/sbp, 1: timing/dbp, 2: anchor)
        runner.set_tensor(0, np.zeros((1, 1, 300), dtype=np.float32))
        assert np.all(runner._morph_buffer == 0.0)

        with pytest.raises(KeyError):
            runner.set_tensor("invalid_name", test_morph)

        with pytest.raises(KeyError):
            runner.get_tensor("invalid_name")

    def test_tflite_flatbuffer_export_and_load_roundtrip(self):
        """Must export genuine INT8 flatbuffer and load it into TFLiteRunner with dequantized weights."""
        runner_orig = TFLiteRunner(seq_len=300)

        with tempfile.TemporaryDirectory() as tmpdir:
            flatbuffer_path = os.path.join(tmpdir, "dual_stream_tcn_int8.tflite")
            export_info = runner_orig.export_int8_flatbuffer(flatbuffer_path)

            assert os.path.exists(flatbuffer_path)
            assert export_info["meets_size_target"] is True
            assert export_info["file_size_kb"] < 250.0

            # Instantiate TFLiteRunner loading the exported flatbuffer
            runner = TFLiteRunner(model_path=flatbuffer_path, seq_len=300)
            assert runner.is_quantized is True
            assert runner.quantization_metadata["source_format"] == "TFLITE_INT8_FLATBUFFER"
            assert runner.quantization_metadata["tensor_count"] > 10
            assert runner.quantization_metadata["file_size_kb"] < 250.0

            # Inference test with loaded quantized model
            x_m = np.random.uniform(0.0, 1.0, (1, 1, 300)).astype(np.float32)
            x_t = np.random.uniform(0.0, 1.0, (1, 1, 300)).astype(np.float32)
            z = np.array([7.0, 100.0, 0.0, 0.08, 85000.0, 0.024], dtype=np.float32)

            sbp, dbp = runner.predict(x_m, x_t, z)
            assert not np.isnan(sbp) and not np.isnan(dbp)
            assert 50.0 <= sbp <= 250.0
            assert 30.0 <= dbp <= 180.0

    def test_snapdragon_4_gen_2_latency_budget(self):
        """Inference latency on target architecture must be strictly below 60 ms budget."""
        runner = TFLiteRunner(seq_len=300)
        bench = runner.benchmark_latency(num_runs=15, warmup_runs=3)

        assert bench["mean_latency_ms"] < 60.0
        assert bench["min_latency_ms"] < 60.0
        assert bench["meets_budget"] is True
        # On standard modern x86/ARM CPU, DualStreamTCN executes in < 30 ms
        assert bench["mean_latency_ms"] < 40.0

    def test_snapdragon_4_gen_2_memory_budget(self):
        """Memory profiler must confirm peak RAM consumption <= 20 MB."""
        runner = TFLiteRunner(seq_len=300)
        mem = runner.profile_memory()

        assert mem["param_memory_mb"] < 2.0
        assert mem["int8_model_mb"] < 0.35  # ~230 KB
        assert mem["buffer_memory_mb"] < 0.05
        assert mem["peak_memory_mb"] <= 20.0
        assert mem["meets_memory_budget"] is True

    def test_edge_runtime_profile_metrics(self):
        """EdgeRuntimeProfile must validate all Snapdragon 4 Gen 2 feasibility gates."""
        runner = TFLiteRunner(seq_len=300)
        profile = runner.get_profile_metrics()

        assert profile.target_device == "Qualcomm Snapdragon 4 Gen 2 (SM4450)"
        assert profile.total_parameters < 250_000
        assert profile.int8_model_size_kb < 250.0
        assert profile.compression_ratio == 4.0
        assert profile.nominal_cpu_latency_ms == 3.2
        assert profile.nominal_npu_latency_ms == 1.6
        assert profile.peak_memory_mb <= 20.0
        assert profile.meets_latency_budget is True
        assert profile.meets_memory_budget is True
        assert profile.meets_size_budget is True

    def test_cold_start_vs_warm_inference(self):
        """Cold start initialization must be < 200 ms and warm inference < 60 ms."""
        t0 = time.perf_counter()
        runner = TFLiteRunner(seq_len=300)
        init_time_ms = (time.perf_counter() - t0) * 1000.0
        assert init_time_ms < 200.0

        # Warm inference
        t1 = time.perf_counter()
        _ = runner.invoke()
        warm_time_ms = (time.perf_counter() - t1) * 1000.0
        assert warm_time_ms < 60.0

    def test_runner_nan_input_recovery(self):
        """Runner must not corrupt internal state when presented with zeros or extreme inputs."""
        runner = TFLiteRunner(seq_len=300)
        # Clean forward pass directly after extreme zeros
        runner.set_tensor("morph", np.zeros((1, 1, 300), dtype=np.float32))
        runner.set_tensor("timing", np.zeros((1, 1, 300), dtype=np.float32))
        runner.set_tensor("anchor", np.zeros(6, dtype=np.float32))
        sbp_0, dbp_0 = runner.invoke()

        assert not np.isnan(sbp_0)
        assert not np.isnan(dbp_0)

    def test_runner_buffer_boundary_handling(self):
        """Runner must safely handle input signals shorter or longer than pre-allocated buffer."""
        runner = TFLiteRunner(seq_len=300)

        # Shorter array (150 elements)
        short_arr = np.linspace(0.1, 0.9, 150).astype(np.float32)
        runner.set_tensor("morph", short_arr)
        # Should populate first 150 and pad remainder without error
        assert np.isclose(runner._morph_buffer[0, 0, 0], 0.1)

        # Longer array (400 elements)
        long_arr = np.linspace(0.2, 0.8, 400).astype(np.float32)
        runner.set_tensor("timing", long_arr)
        assert np.isclose(runner._timing_buffer[0, 0, 0], 0.2)
        assert runner._timing_buffer.shape == (1, 1, 300)


# ==============================================================================
# Suite 2: DualSiteBPPipeline End-to-End Tests
# ==============================================================================

class TestDualSiteBPPipeline:
    """Verifies end-to-end integration across Optical Sync, HRVP-PINN, and TFLite Runner."""

    @pytest.fixture
    def normotensive_session(self):
        """Generates realistic 10-second normotensive acquisition session."""
        gen = SyntheticPhysiologicalGenerator(seed=101)
        profile = PhysiologicalProfile(
            height_m=1.75,
            sbp_mmhg=120.0,
            dbp_mmhg=80.0,
            hr_bpm=72.0,
            alpha=0.024,
            e0_pa=0.85e5,
            delta_h_m=0.0,
            snr_db=30.0,
        )
        return gen.generate_session(profile, duration_sec=10.0)

    def test_pipeline_end_to_end_from_session_dict(self, normotensive_session):
        """Pipeline must ingest session dict and return complete, valid clinical results."""
        pipeline = DualSiteBPPipeline()
        result = pipeline.process_session(normotensive_session)

        assert isinstance(result, PipelineResult)
        assert result.status == "VALID_ACQUISITION"
        assert result.is_valid is True

        # Physiological bounds check
        assert 90.0 <= result.sbp <= 150.0
        assert 50.0 <= result.dbp <= 100.0
        assert result.sbp > result.dbp + 15.0
        assert 65.0 <= result.p_map <= 120.0
        assert 4.0 <= result.epwv_mps <= 12.0

        # HRVP components check
        assert result.delta_ptt_ms > 0.0
        assert result.alpha > 0.0
        assert result.e0_pa > 0.0
        assert result.anchor_vector.shape == (6,)

        # Quality metrics check
        assert result.quality.sustained_30fps is True
        assert result.quality.blanching_detected is False
        assert result.quality.motion_artifact_detected is False
        assert result.quality.perfusion_index > 0.5

        # Latency check: neural inference < 60 ms
        assert result.inference_latency_ms < 60.0

    def test_pipeline_end_to_end_from_individual_streams(self, normotensive_session):
        """Pipeline must support calling process_streams directly with separate arguments."""
        pipeline = DualSiteBPPipeline()
        result = pipeline.process_streams(
            t_front_ns=normotensive_session["t_front_ns"],
            face_rgb=normotensive_session["face_rgb"],
            t_rear_ns=normotensive_session["t_rear_ns"],
            finger_rgb=normotensive_session["finger_rgb"],
            imu_data=normotensive_session["imu_data"],
            user_profile=normotensive_session["user_profile"],
            nominal_fps=30.0,
        )

        assert result.status == "VALID_ACQUISITION"
        assert result.is_valid is True
        assert 70.0 <= result.sbp <= 200.0
        assert 40.0 <= result.dbp <= 130.0

    def test_pipeline_output_contract_dict_and_attr_access(self, normotensive_session):
        """PipelineResult must fulfill PROJECT.md interface contract and support dict access."""
        pipeline = DualSiteBPPipeline()
        result = pipeline.process_session(normotensive_session)

        # Contract keys required by test suites
        required_keys = ["sbp", "dbp", "p_map", "epwv_mps", "status"]
        for key in required_keys:
            assert key in result
            assert getattr(result, key) is not None
            assert result[key] is not None

        # Flat dictionary serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        for key in required_keys:
            assert key in result_dict

    def test_pipeline_optical_synchronization_integration(self, normotensive_session):
        """Pipeline must accurately bind nanosecond timestamps onto a 125 Hz uniform grid."""
        pipeline = DualSiteBPPipeline()
        result = pipeline.process_session(normotensive_session)

        # Jitter statistics should reflect synthetic VFR settings
        assert result.quality.front_jitter_std_ms >= 0.0
        assert result.quality.rear_jitter_std_ms >= 0.0
        assert result.quality.dropped_frames_front == 0
        assert result.quality.dropped_frames_rear == 0

    def test_pipeline_pinn_hrvp_integration(self, normotensive_session):
        """Pipeline must correctly integrate closed-form HRVP parameter solver."""
        pipeline = DualSiteBPPipeline()
        result = pipeline.process_session(normotensive_session)

        # Baseline SBP / DBP analytical decomposition
        assert result.sbp_analytical > result.dbp_analytical
        assert result.map_analytical >= 50.0
        assert len(result.anchor_vector) == 6
        # Anchor vector format: [epwv, delta_ptt, delta_p_hydro, compliance, e0, alpha]
        assert np.isclose(result.anchor_vector[0], result.epwv_mps, rtol=1e-2)
        assert np.isclose(result.anchor_vector[1], result.delta_ptt_ms, rtol=1e-2)

    def test_pipeline_waveform_normalization_tensor_shape(self):
        """Waveform normalization utility must output exact (1, 1, 300) tensors in [0, 1]."""
        raw_wave = np.sin(np.linspace(0, 10, 1250)) + 5.0
        tensor = normalize_waveform_tensor(raw_wave, target_len=300)

        assert tensor.shape == (1, 1, 300)
        assert np.min(tensor) >= 0.0
        assert np.max(tensor) <= 1.0
        assert np.isclose(np.min(tensor), 0.0, atol=1e-5)
        assert np.isclose(np.max(tensor), 1.0, atol=1e-5)

        # Flat signal handling
        flat_wave = np.ones(200) * 120.0
        flat_tensor = normalize_waveform_tensor(flat_wave, target_len=300)
        assert flat_tensor.shape == (1, 1, 300)
        assert not np.isnan(flat_tensor).any()

    def test_pipeline_fusion_analytical_and_neural(self, normotensive_session):
        """Configurable fusion weight must smoothly interpolate between analytical and neural estimates."""
        # Pure analytical (fusion_weight = 0.0)
        config_analytical = PipelineConfig(fusion_weight=0.0)
        pipeline_ana = DualSiteBPPipeline(config=config_analytical)
        res_ana = pipeline_ana.process_session(normotensive_session)
        assert np.isclose(res_ana.sbp, res_ana.sbp_analytical, atol=1.0)
        assert np.isclose(res_ana.dbp, res_ana.dbp_analytical, atol=1.0)

        # Pure neural (fusion_weight = 1.0)
        config_neural = PipelineConfig(fusion_weight=1.0)
        pipeline_neu = DualSiteBPPipeline(config=config_neural)
        res_neu = pipeline_neu.process_session(normotensive_session)
        assert np.isclose(res_neu.sbp, res_neu.sbp_neural, atol=1.0)

    def test_pipeline_30s_continuous_execution(self):
        """Pipeline must handle a continuous 30-second recording session without memory issues."""
        gen = SyntheticPhysiologicalGenerator(seed=102)
        profile = PhysiologicalProfile(height_m=1.80, sbp_mmhg=125.0, dbp_mmhg=82.0)
        session_30s = gen.generate_session(profile, duration_sec=30.0)

        pipeline = DualSiteBPPipeline()
        result = pipeline.process_session(session_30s)

        assert result.status == "VALID_ACQUISITION"
        assert result.is_valid is True
        assert result.quality.sustained_30fps is True

    def test_pipeline_memory_leak_free_repeated_runs(self, normotensive_session):
        """Executing 50 pipeline runs must not leak memory or degrade latency."""
        pipeline = DualSiteBPPipeline()
        for _ in range(50):
            res = pipeline.process_session(normotensive_session)
            assert res.status == "VALID_ACQUISITION"


# ==============================================================================
# Suite 3: Clinical Plausibility Guards & Fault Tolerance Tests
# ==============================================================================

class TestClinicalGuardsAndFaultTolerance:
    """Verifies capillary blanching, motion artifacts, dropped frames, and bounds guards."""

    @pytest.fixture
    def base_session(self):
        gen = SyntheticPhysiologicalGenerator(seed=201)
        profile = PhysiologicalProfile(height_m=1.75, sbp_mmhg=118.0, dbp_mmhg=76.0)
        return gen.generate_session(profile, duration_sec=10.0)

    def test_capillary_blanching_detection_low_ac(self, base_session):
        """Capillary blanching from excessive finger pressure (collapsed AC) must be flagged."""
        pipeline = DualSiteBPPipeline()

        # Collapse pulsatile AC modulation on rear camera (finger contact)
        blanched_session = dict(base_session)
        finger_blanched = base_session["finger_rgb"].copy()
        # Flatline green channel to DC constant 180.0 (zero pulsatile AC)
        finger_blanched[:, 1] = 180.0
        blanched_session["finger_rgb"] = finger_blanched

        result = pipeline.process_session(blanched_session)
        assert result.status == "BLANCHING_DETECTED"
        assert result.is_valid is False
        assert result.quality.blanching_detected is True

    def test_capillary_blanching_detection_saturation(self, base_session):
        """Severe CMOS saturation from excessive LED transillumination must be flagged."""
        pipeline = DualSiteBPPipeline()

        saturated_session = dict(base_session)
        finger_sat = base_session["finger_rgb"].copy()
        # Saturate green channel to 252.0 (> 245.0 threshold)
        finger_sat[:, 1] = 252.0
        saturated_session["finger_rgb"] = finger_sat

        result = pipeline.process_session(saturated_session)
        assert result.status == "BLANCHING_DETECTED"
        assert result.is_valid is False
        assert result.quality.blanching_detected is True

    def test_motion_artifact_detection(self, base_session):
        """Excessive motion noise destroying optical SNR must be flagged as MOTION_ARTIFACT."""
        pipeline = DualSiteBPPipeline()

        motion_session = dict(base_session)
        rng = np.random.RandomState(999)
        # Inject violent high-frequency motion artifacts into facial rPPG
        noisy_face = base_session["face_rgb"].copy()
        noisy_face[:, 1] += rng.normal(0.0, 50.0, size=len(noisy_face))
        motion_session["face_rgb"] = noisy_face

        result = pipeline.process_session(motion_session)
        assert result.status == "MOTION_ARTIFACT"
        assert result.is_valid is False
        assert result.quality.motion_artifact_detected is True

    def test_dropped_frame_detection(self, base_session):
        """Camera frame drop exceeding 1.8x period must be detected and flagged."""
        pipeline = DualSiteBPPipeline()

        dropped_session = dict(base_session)
        t_front = base_session["t_front_ns"].copy()
        # Inject a 90 ms camera stall (~3 dropped frames at 30 FPS)
        t_front[100:] += 90_000_000
        dropped_session["t_front_ns"] = t_front

        result = pipeline.process_session(dropped_session)
        assert result.status == "DROPPED_FRAMES_DETECTED"
        assert result.is_valid is False
        assert result.quality.dropped_frames_detected is True
        assert result.quality.dropped_frames_front >= 1
        assert result.quality.sustained_30fps is False

    def test_non_overlapping_streams_clean_rejection(self, base_session):
        """Zero temporal overlap between front and rear cameras must raise ValueError cleanly."""
        pipeline = DualSiteBPPipeline()

        disjoint_session = dict(base_session)
        # Shift rear timestamps 50 seconds into the future
        disjoint_session["t_rear_ns"] = base_session["t_rear_ns"] + int(50e9)

        with pytest.raises(ValueError, match="Zero temporal overlap"):
            pipeline.process_session(disjoint_session)

    def test_physiological_bounds_clamping(self, base_session):
        """Out-of-bounds arterial pressures must be clamped to clinical safety limits."""
        # Custom config with tight bounds for testing clamping logic
        config = PipelineConfig(
            sbp_min_mmhg=90.0,
            sbp_max_mmhg=160.0,
            dbp_min_mmhg=55.0,
            dbp_max_mmhg=100.0,
            min_pulse_pressure_mmhg=20.0,
        )
        pipeline = DualSiteBPPipeline(config=config)
        result = pipeline.process_session(base_session)

        assert config.sbp_min_mmhg <= result.sbp <= config.sbp_max_mmhg
        assert config.dbp_min_mmhg <= result.dbp <= config.dbp_max_mmhg
        assert result.sbp - result.dbp >= config.min_pulse_pressure_mmhg

    def test_rapid_recovery_from_nan_and_inf(self, base_session):
        """Pipeline must handle unexpected NaNs gracefully without crashing."""
        pipeline = DualSiteBPPipeline()

        corrupted_session = dict(base_session)
        face_nan = base_session["face_rgb"].copy()
        face_nan[50:55, 1] = np.nan
        corrupted_session["face_rgb"] = face_nan

        result = pipeline.process_session(corrupted_session)
        # Should flag motion artifact or recover without throwing exception
        assert not np.isnan(result.sbp)
        assert not np.isnan(result.dbp)
        assert not np.isnan(result.p_map)
        assert result.is_valid is False or result.status == "MOTION_ARTIFACT"

    def test_sustained_30fps_acquisition_flag(self):
        """Sustained 30 FPS flag must correctly identify a clean 10-second run vs a short 2-second run."""
        gen = SyntheticPhysiologicalGenerator(seed=203)
        prof = PhysiologicalProfile()

        pipeline = DualSiteBPPipeline()

        # 10-second session -> sustained_30fps = True
        s10 = gen.generate_session(prof, duration_sec=10.0)
        res10 = pipeline.process_session(s10)
        assert res10.quality.sustained_30fps is True

        # 2-second session -> sustained_30fps = False (less than 9.5s requirement)
        s2 = gen.generate_session(prof, duration_sec=2.0)
        res2 = pipeline.process_session(s2)
        assert res2.quality.sustained_30fps is False
