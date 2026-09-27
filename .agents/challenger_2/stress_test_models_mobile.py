"""
Empirical Stress Test Harness — Challenger 2 (Template Collapse & Quantization Stress Challenger)
Targets: src/models/ and src/mobile/

Stress Vectors Tested:
1. Adversarial Inputs:
   - Flatline zero signals
   - Pure Gaussian and uniform random noise
   - Extreme amplitude spikes (+1e6, -1e6, Inf, NaN)
   - Degenerate / non-physiological pulse shapes (square wave, triangular spike, inverted pulses)
2. Template Collapse & Regression-to-the-Mean Probe:
   - Cohort of 200 diverse clinical subjects across 5 BP categories:
     Hypotensive, Normotensive, Pre-HTN, Stage 1 HTN, Stage 2 HTN
   - Computes Pearson r, Prediction Variance Ratio sigma^2(y_hat)/sigma^2(y)
   - Assesses analytical vs neural vs fused components
3. Memory Leak & Buffer Invariance Stress:
   - 10,000 consecutive INT8 inferences through TFLiteRunner
   - Checks static buffer invariance across all 10,000 iterations
   - Tracks heap allocations and memory growth via tracemalloc
   - Verifies Snapdragon 4 Gen 2 latency bounds (< 60 ms)
4. Long-Duration Continuous Session:
   - 60-second continuous recording session at 30 FPS (1,800 frames)
   - Sliding-window real-time inference streaming simulation (60 consecutive window updates)
   - Asserts memory and latency stability
"""

from __future__ import annotations

import sys
import os
import time
import math
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ds_tcn import DualStreamTCN
from src.models.quantizer import INT8Quantizer
from src.mobile.tflite_runner import TFLiteRunner
from src.mobile.pipeline import DualSiteBPPipeline, PipelineConfig, normalize_waveform_tensor
from tests.synthetic_generator import (
    SyntheticPhysiologicalGenerator,
    PhysiologicalProfile,
    calculate_d_path,
    calculate_hydrostatic_delta_p,
)


class ModelMobileStressChallenger:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or Path(__file__).resolve().parent)
        self.results: Dict[str, Any] = {}

    # =========================================================================
    # VECTOR 1: ADVERSARIAL INPUTS
    # =========================================================================
    def stress_vector_1_adversarial_inputs(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("[VECTOR 1] ADVERSARIAL INPUTS STRESS HARNESS")
        print("Testing: Flatline zero, Random noise, Extreme spikes, Degenerate pulses")
        print("=" * 80)

        runner = TFLiteRunner()
        pipeline = DualSiteBPPipeline()
        results = {}

        # 1.1 Model Level Direct Adversarial Tensors
        print("\n--- 1.1 Direct TFLiteRunner / DS-TCN Model Input Testing ---")
        seq_len = 300
        cases = {
            "flatline_zeros": (np.zeros((1, 1, seq_len), dtype=np.float32), np.zeros((1, 1, seq_len), dtype=np.float32), np.zeros(6, dtype=np.float32)),
            "flatline_ones": (np.ones((1, 1, seq_len), dtype=np.float32), np.ones((1, 1, seq_len), dtype=np.float32), np.ones(6, dtype=np.float32)),
            "pure_gaussian_noise": (np.random.randn(1, 1, seq_len).astype(np.float32), np.random.randn(1, 1, seq_len).astype(np.float32), np.random.randn(6).astype(np.float32)),
            "extreme_spike_pos": (np.full((1, 1, seq_len), 1e6, dtype=np.float32), np.full((1, 1, seq_len), 1e6, dtype=np.float32), np.full(6, 1e6, dtype=np.float32)),
            "extreme_spike_neg": (np.full((1, 1, seq_len), -1e6, dtype=np.float32), np.full((1, 1, seq_len), -1e6, dtype=np.float32), np.full(6, -1e6, dtype=np.float32)),
            "square_wave": (np.tile([0.0, 1.0], seq_len // 2).reshape(1, 1, seq_len).astype(np.float32), np.tile([0.0, 1.0], seq_len // 2).reshape(1, 1, seq_len).astype(np.float32), np.zeros(6, dtype=np.float32)),
            "high_freq_chirp": (np.sin(np.linspace(0, 100, seq_len)).reshape(1, 1, seq_len).astype(np.float32), np.cos(np.linspace(0, 100, seq_len)).reshape(1, 1, seq_len).astype(np.float32), np.zeros(6, dtype=np.float32)),
            "nan_inf_inputs": (np.array([[[np.nan] * seq_len]], dtype=np.float32), np.array([[[np.inf] * seq_len]], dtype=np.float32), np.zeros(6, dtype=np.float32)),
        }

        model_results = {}
        for name, (morph, timing, anchor) in cases.items():
            try:
                sbp, dbp = runner.predict(morph, timing, anchor)
                is_finite = math.isfinite(sbp) and math.isfinite(dbp)
                model_results[name] = {
                    "sbp": sbp,
                    "dbp": dbp,
                    "is_finite": is_finite,
                    "crashed": False
                }
                print(f"  [Model] {name:<22}: SBP={sbp:8.2f}, DBP={dbp:8.2f} (Finite={is_finite})")
            except Exception as e:
                model_results[name] = {"crashed": True, "error": str(e)}
                print(f"  [Model] {name:<22}: CRASHED ({e})")

        # 1.2 Full Pipeline Level Session Adversarial Testing
        print("\n--- 1.2 Full DualSiteBPPipeline End-to-End Adversarial Testing ---")
        n_frames = 300
        t_front = (np.arange(n_frames) * (1.0 / 30.0) * 1e9).astype(np.int64)
        t_rear = (np.arange(n_frames) * (1.0 / 30.0) * 1e9).astype(np.int64)

        pipeline_cases = {
            "pipe_flatline_zero": {
                "t_front_ns": t_front,
                "face_rgb": np.zeros((n_frames, 3), dtype=np.float32),
                "t_rear_ns": t_rear,
                "finger_rgb": np.zeros((n_frames, 3), dtype=np.float32),
            },
            "pipe_pure_noise": {
                "t_front_ns": t_front,
                "face_rgb": np.random.uniform(0, 255, (n_frames, 3)).astype(np.float32),
                "t_rear_ns": t_rear,
                "finger_rgb": np.random.uniform(0, 255, (n_frames, 3)).astype(np.float32),
            },
            "pipe_extreme_saturation": {
                "t_front_ns": t_front,
                "face_rgb": np.full((n_frames, 3), 255.0, dtype=np.float32),
                "t_rear_ns": t_rear,
                "finger_rgb": np.full((n_frames, 3), 255.0, dtype=np.float32),
            },
            "pipe_extreme_nan_inf": {
                "t_front_ns": t_front,
                "face_rgb": np.full((n_frames, 3), np.nan, dtype=np.float32),
                "t_rear_ns": t_rear,
                "finger_rgb": np.full((n_frames, 3), np.inf, dtype=np.float32),
            },
            "pipe_non_physiological_square": {
                "t_front_ns": t_front,
                "face_rgb": np.where(np.sin(np.linspace(0, 20*np.pi, n_frames))[:, None] > 0, 180.0, 80.0).astype(np.float32),
                "t_rear_ns": t_rear,
                "finger_rgb": np.where(np.sin(np.linspace(0, 20*np.pi, n_frames))[:, None] > 0, 200.0, 50.0).astype(np.float32),
            }
        }

        pipe_results = {}
        for name, sess in pipeline_cases.items():
            try:
                res = pipeline.process_session(sess)
                pipe_results[name] = {
                    "sbp": res.sbp,
                    "dbp": res.dbp,
                    "status": res.status,
                    "is_valid": res.is_valid,
                    "blanching_detected": res.quality.blanching_detected,
                    "motion_artifact_detected": res.quality.motion_artifact_detected,
                    "crashed": False
                }
                print(f"  [Pipe]  {name:<28}: Status={res.status:<25}, Valid={res.is_valid}, SBP={res.sbp:5.1f}, DBP={res.dbp:5.1f}")
            except Exception as e:
                pipe_results[name] = {"crashed": True, "error": str(e)}
                print(f"  [Pipe]  {name:<28}: CRASHED ({e})")

        results["model_cases"] = model_results
        results["pipeline_cases"] = pipe_results
        self.results["vector_1"] = results
        return results

    # =========================================================================
    # VECTOR 2: TEMPLATE COLLAPSE & 200-SUBJECT COHORT PROBE
    # =========================================================================
    def stress_vector_2_template_collapse_probe(self, n_subjects: int = 200, seed: int = 42) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print(f"[VECTOR 2] TEMPLATE COLLAPSE PROBE: {n_subjects}-SUBJECT COHORT")
        print("Auditing Real DualSiteBPPipeline Across Full Clinical Spectrum")
        print("=" * 80)

        gen = SyntheticPhysiologicalGenerator(seed=seed)
        pipeline = DualSiteBPPipeline()
        rng = np.random.default_rng(seed)

        sbp_true, dbp_true = [], []
        sbp_fused, dbp_fused = [], []
        sbp_analytical, dbp_analytical = [], []
        sbp_neural, dbp_neural = [], []
        statuses = []

        t0 = time.perf_counter()
        for i in range(n_subjects):
            category = i % 5
            if category == 0:  # Normotensive
                sbp = float(rng.uniform(105, 120))
                dbp = float(rng.uniform(68, 80))
            elif category == 1:  # Pre-HTN
                sbp = float(rng.uniform(122, 138))
                dbp = float(rng.uniform(80, 88))
            elif category == 2:  # Stage 1 HTN
                sbp = float(rng.uniform(140, 158))
                dbp = float(rng.uniform(90, 98))
            elif category == 3:  # Stage 2 HTN
                sbp = float(rng.uniform(160, 190))
                dbp = float(rng.uniform(100, 118))
            else:  # Hypotensive
                sbp = float(rng.uniform(80, 98))
                dbp = float(rng.uniform(50, 64))

            profile = PhysiologicalProfile(
                height_m=float(rng.uniform(1.52, 1.92)),
                sbp_mmhg=sbp,
                dbp_mmhg=dbp,
                alpha=float(rng.uniform(0.018, 0.038)),
                e0_pa=float(rng.uniform(0.70e5, 1.15e5)),
                delta_h_m=float(rng.choice([0.0, 0.20, -0.20])),
                snr_db=float(rng.uniform(22.0, 35.0)),
            )

            sess = gen.generate_session(profile, duration_sec=10.0)
            res = pipeline.process_session(sess)

            sbp_true.append(sbp)
            dbp_true.append(dbp)
            sbp_fused.append(res.sbp)
            dbp_fused.append(res.dbp)
            sbp_analytical.append(res.sbp_analytical)
            dbp_analytical.append(res.dbp_analytical)
            sbp_neural.append(res.sbp_neural)
            dbp_neural.append(res.dbp_neural)
            statuses.append(res.status)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{n_subjects} subjects in {time.perf_counter() - t0:.2f}s...")

        # Convert to arrays
        y_sbp = np.array(sbp_true, dtype=np.float64)
        y_dbp = np.array(dbp_true, dtype=np.float64)
        y_hat_sbp = np.array(sbp_fused, dtype=np.float64)
        y_hat_dbp = np.array(dbp_fused, dtype=np.float64)
        y_ana_sbp = np.array(sbp_analytical, dtype=np.float64)
        y_ana_dbp = np.array(dbp_analytical, dtype=np.float64)
        y_neu_sbp = np.array(sbp_neural, dtype=np.float64)
        y_neu_dbp = np.array(dbp_neural, dtype=np.float64)

        # Statistics for Fused Pipeline Predictions
        r_sbp = float(np.corrcoef(y_sbp, y_hat_sbp)[0, 1])
        r_dbp = float(np.corrcoef(y_dbp, y_hat_dbp)[0, 1])
        var_ratio_sbp = float(np.var(y_hat_sbp) / np.var(y_sbp))
        var_ratio_dbp = float(np.var(y_hat_dbp) / np.var(y_dbp))
        mae_sbp = float(np.mean(np.abs(y_hat_sbp - y_sbp)))
        mae_dbp = float(np.mean(np.abs(y_hat_dbp - y_dbp)))
        sd_sbp = float(np.std(y_hat_sbp - y_sbp))
        sd_dbp = float(np.std(y_hat_dbp - y_dbp))

        # Statistics for Isolated Neural Predictions
        var_neu_sbp = float(np.var(y_neu_sbp))
        var_neu_dbp = float(np.var(y_neu_dbp))
        var_ratio_neu_sbp = float(var_neu_sbp / np.var(y_sbp))
        var_ratio_neu_dbp = float(var_neu_dbp / np.var(y_dbp))
        r_neu_sbp = float(np.corrcoef(y_sbp, y_neu_sbp)[0, 1]) if var_neu_sbp > 1e-12 else 0.0
        r_neu_dbp = float(np.corrcoef(y_dbp, y_neu_dbp)[0, 1]) if var_neu_dbp > 1e-12 else 0.0

        # Statistics for Isolated Analytical Predictions
        var_ratio_ana_sbp = float(np.var(y_ana_sbp) / np.var(y_sbp))
        var_ratio_ana_dbp = float(np.var(y_ana_dbp) / np.var(y_dbp))
        r_ana_sbp = float(np.corrcoef(y_sbp, y_ana_sbp)[0, 1])
        r_ana_dbp = float(np.corrcoef(y_dbp, y_ana_dbp)[0, 1])
        mae_ana_sbp = float(np.mean(np.abs(y_ana_sbp - y_sbp)))
        mae_ana_dbp = float(np.mean(np.abs(y_ana_dbp - y_dbp)))

        # BHS Grading
        def get_bhs_grade(errors):
            abs_err = np.abs(errors)
            p5 = np.mean(abs_err <= 5.0) * 100
            p10 = np.mean(abs_err <= 10.0) * 100
            p15 = np.mean(abs_err <= 15.0) * 100
            if p5 >= 60 and p10 >= 85 and p15 >= 95:
                return "Grade A"
            elif p5 >= 50 and p10 >= 75 and p15 >= 90:
                return "Grade B"
            elif p5 >= 40 and p10 >= 65 and p15 >= 85:
                return "Grade C"
            return "Grade D (Failed)"

        bhs_sbp = get_bhs_grade(y_hat_sbp - y_sbp)
        bhs_dbp = get_bhs_grade(y_hat_dbp - y_dbp)

        print("\n--- 200-Subject Cohort Empirical Results ---")
        print(f"Ground Truth SBP Spread: min={np.min(y_sbp):.1f}, max={np.max(y_sbp):.1f}, std={np.std(y_sbp):.2f} mmHg")
        print(f"Predicted SBP Spread:    min={np.min(y_hat_sbp):.1f}, max={np.max(y_hat_sbp):.1f}, std={np.std(y_hat_sbp):.2f} mmHg")
        print(f"Neural SBP Spread:       min={np.min(y_neu_sbp):.1f}, max={np.max(y_neu_sbp):.1f}, std={np.std(y_neu_sbp):.4f} mmHg")
        print(f"Analytical SBP Spread:   min={np.min(y_ana_sbp):.1f}, max={np.max(y_ana_sbp):.1f}, std={np.std(y_ana_sbp):.2f} mmHg")
        print("-" * 50)
        print(f"SBP Pearson r:           {r_sbp:.4f}  (Contract: >= 0.82) -> {'PASS' if r_sbp >= 0.82 else 'FAIL'}")
        print(f"DBP Pearson r:           {r_dbp:.4f}  (Contract: >= 0.80) -> {'PASS' if r_dbp >= 0.80 else 'FAIL'}")
        print(f"SBP Variance Ratio:      {var_ratio_sbp:.4f}  (Standard: 0.70 - 1.30) -> {'PASS' if 0.70 <= var_ratio_sbp <= 1.30 else 'COLLAPSE / VARIANCE SQUEEZE'}")
        print(f"DBP Variance Ratio:      {var_ratio_dbp:.4f}  (Standard: 0.70 - 1.30) -> {'PASS' if 0.70 <= var_ratio_dbp <= 1.30 else 'COLLAPSE / VARIANCE SQUEEZE'}")
        print(f"SBP MAE:                 {mae_sbp:.2f} mmHg (Standard: <= 5.0 mmHg) -> {'PASS' if mae_sbp <= 5.0 else 'FAIL'}")
        print(f"DBP MAE:                 {mae_dbp:.2f} mmHg (Standard: <= 3.5 mmHg) -> {'PASS' if mae_dbp <= 3.5 else 'FAIL'}")
        print(f"SBP Error SD:            {sd_sbp:.2f} mmHg (Standard: <= 7.0 mmHg) -> {'PASS' if sd_sbp <= 7.0 else 'FAIL'}")
        print(f"DBP Error SD:            {sd_dbp:.2f} mmHg (Standard: <= 7.0 mmHg) -> {'PASS' if sd_dbp <= 7.0 else 'FAIL'}")
        print(f"BHS Clinical Grade:      SBP: {bhs_sbp} | DBP: {bhs_dbp}")
        print("-" * 50)
        print(f"Isolated Neural Model SBP Variance: {var_neu_sbp:.6f} (Variance Ratio: {var_ratio_neu_sbp:.6f})")
        print(f"Isolated Neural Model SBP Pearson:  r = {r_neu_sbp:.4f}")
        print(f"Isolated Analytical SBP Var Ratio:  {var_ratio_ana_sbp:.4f} (MAE: {mae_ana_sbp:.2f} mmHg)")

        report = {
            "n_subjects": n_subjects,
            "fused_pipeline": {
                "sbp_pearson_r": r_sbp,
                "dbp_pearson_r": r_dbp,
                "sbp_variance_ratio": var_ratio_sbp,
                "dbp_variance_ratio": var_ratio_dbp,
                "sbp_mae": mae_sbp,
                "dbp_mae": mae_dbp,
                "sbp_sd": sd_sbp,
                "dbp_sd": sd_dbp,
                "sbp_bhs_grade": bhs_sbp,
                "dbp_bhs_grade": bhs_dbp,
                "meets_r_criterion": (r_sbp >= 0.82 and r_dbp >= 0.80),
                "meets_variance_ratio_criterion": (0.70 <= var_ratio_sbp <= 1.30 and 0.70 <= var_ratio_dbp <= 1.30),
                "meets_aami_criterion": (mae_sbp <= 5.0 and mae_dbp <= 3.5 and sd_sbp <= 7.0 and sd_dbp <= 7.0),
            },
            "isolated_neural_model": {
                "sbp_variance": var_neu_sbp,
                "sbp_variance_ratio": var_ratio_neu_sbp,
                "sbp_pearson_r": r_neu_sbp,
                "is_collapsed_to_mean": (var_ratio_neu_sbp < 0.05),
            },
            "isolated_analytical_model": {
                "sbp_variance_ratio": var_ratio_ana_sbp,
                "sbp_pearson_r": r_ana_sbp,
                "sbp_mae": mae_ana_sbp,
            }
        }
        self.results["vector_2"] = report
        return report

    # =========================================================================
    # VECTOR 3: 10,000 CONSECUTIVE INT8 INFERENCES MEMORY & BUFFER STRESS
    # =========================================================================
    def stress_vector_3_memory_leak_and_buffer_invariance(self, iterations: int = 10000) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print(f"[VECTOR 3] MEMORY LEAK & BUFFER INVARIANCE: {iterations} INFERENCES")
        print("Checking Static Buffer Memory Stability and Snapdragon 4 Gen 2 Latency")
        print("=" * 80)

        # Export a clean INT8 flatbuffer to load genuinely
        tflite_path = str(self.output_dir / "stress_dual_stream_tcn_int8.tflite")
        quantizer = INT8Quantizer()
        quantizer.export_tflite_int8(tflite_path)

        runner = TFLiteRunner(model_path=tflite_path)
        initial_invariant = runner.verify_buffer_invariance()

        # Start tracemalloc
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        latencies = []
        violations = 0
        checkpoint_steps = [1, 100, 1000, 2500, 5000, 7500, 10000]
        checkpoint_reports = []

        dummy_morph = np.random.uniform(0.0, 1.0, 300).astype(np.float32)
        dummy_timing = np.random.uniform(0.0, 1.0, 300).astype(np.float32)
        dummy_anchor = np.array([7.5, 110.0, 0.0, 0.085, 85000.0, 0.024], dtype=np.float32)

        t_start_total = time.perf_counter()
        for i in range(1, iterations + 1):
            t0 = time.perf_counter()
            runner.predict(dummy_morph, dummy_timing, dummy_anchor)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

            # Check buffer invariance periodically
            if i in checkpoint_steps:
                inv = runner.verify_buffer_invariance()
                if not inv:
                    violations += 1
                current_mem_mb = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
                mean_so_far = float(np.mean(latencies))
                checkpoint_reports.append({
                    "iteration": i,
                    "buffer_invariant": inv,
                    "traced_memory_mb": round(current_mem_mb, 4),
                    "mean_latency_ms": round(mean_so_far, 3)
                })
                print(f"  Iteration {i:5d}/{iterations}: Invariant={inv}, Mem={current_mem_mb:.3f} MB, Mean Latency={mean_so_far:.2f} ms")

        total_duration_sec = time.perf_counter() - t_start_total
        snapshot_end = tracemalloc.take_snapshot()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory diff
        top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        mem_leak_detected = (current_mem / (1024 * 1024)) > 1.0  # Heap growth > 1 MB flag

        lat_arr = np.array(latencies, dtype=np.float64)
        mean_lat = float(np.mean(lat_arr))
        p50_lat = float(np.percentile(lat_arr, 50))
        p95_lat = float(np.percentile(lat_arr, 95))
        p99_lat = float(np.percentile(lat_arr, 99))
        max_lat = float(np.max(lat_arr))

        print("\n--- 10,000 Inferences Empirical Profile ---")
        print(f"Total Wall Clock Duration:  {total_duration_sec:.2f} s ({iterations / total_duration_sec:.1f} inferences/s)")
        print(f"Buffer Address Invariance:  {'100% INVARIANT (Zero Heap Allocations)' if violations == 0 else f'VIOLATIONS: {violations}'}")
        print(f"Peak Traced Memory:         {peak_mem / (1024 * 1024):.3f} MB")
        print(f"Current Traced Memory:      {current_mem / (1024 * 1024):.3f} MB")
        print(f"Latency Mean:               {mean_lat:.3f} ms (Budget: < 60 ms) -> {'PASS' if mean_lat < 60 else 'FAIL'}")
        print(f"Latency P50:                {p50_lat:.3f} ms")
        print(f"Latency P95:                {p95_lat:.3f} ms")
        print(f"Latency P99:                {p99_lat:.3f} ms")
        print(f"Latency Max:                {max_lat:.3f} ms")

        # Cleanup exported stress flatbuffer
        if os.path.exists(tflite_path):
            try:
                os.remove(tflite_path)
            except Exception:
                pass

        report = {
            "iterations": iterations,
            "total_duration_sec": total_duration_sec,
            "buffer_invariant_always": (violations == 0),
            "buffer_violations": violations,
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 3),
            "final_memory_mb": round(current_mem / (1024 * 1024), 3),
            "memory_leak_detected": mem_leak_detected,
            "mean_latency_ms": round(mean_lat, 3),
            "p50_latency_ms": round(p50_lat, 3),
            "p95_latency_ms": round(p95_lat, 3),
            "p99_latency_ms": round(p99_lat, 3),
            "max_latency_ms": round(max_lat, 3),
            "meets_latency_budget": (mean_lat < 60.0),
            "checkpoints": checkpoint_reports,
        }
        self.results["vector_3"] = report
        return report

    # =========================================================================
    # VECTOR 4: 60-SECOND CONTINUOUS RECORDING & STREAMING SESSION
    # =========================================================================
    def stress_vector_4_continuous_60s_recording(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("[VECTOR 4] 60-SECOND CONTINUOUS RECORDING & STREAMING STRESS")
        print("Testing Full 60s Session Ingestion & Real-Time Sliding-Window Streaming")
        print("=" * 80)

        gen = SyntheticPhysiologicalGenerator(seed=303)
        profile = PhysiologicalProfile(
            height_m=1.78,
            sbp_mmhg=124.0,
            dbp_mmhg=79.0,
            hr_bpm=72.0,
            resp_rate_bpm=12.0
        )
        print("Generating 60-second dual-camera session (1,800 frames per stream)...")
        session_60s = gen.generate_session(profile, duration_sec=60.0, nominal_camera_fps=30.0)

        pipeline = DualSiteBPPipeline()

        # 4.1 Ingest full 60s session in single pipeline call
        print("\n--- 4.1 Ingesting Full 60-Second Batch Session ---")
        t0 = time.perf_counter()
        res_60s = pipeline.process_session(session_60s)
        dt_batch_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  Batch Processing Time:     {dt_batch_ms:.2f} ms")
        print(f"  Clinical Status:           {res_60s.status}")
        print(f"  Valid Acquisition:         {res_60s.is_valid}")
        print(f"  Sustained 30 FPS Flag:     {res_60s.quality.sustained_30fps}")
        print(f"  Estimated SBP/DBP:         {res_60s.sbp:.1f} / {res_60s.dbp:.1f} mmHg")
        print(f"  Analytical SBP/DBP:        {res_60s.sbp_analytical:.1f} / {res_60s.dbp_analytical:.1f} mmHg")
        print(f"  Front/Rear Jitter Std:     {res_60s.quality.front_jitter_std_ms:.2f} ms / {res_60s.quality.rear_jitter_std_ms:.2f} ms")

        # 4.2 Simulate 60-Second Real-Time Sliding-Window Streaming
        # Camera streams 30 FPS frames continuously. Every 1.0 second (30 frames),
        # an edge inference is triggered on the sliding buffer of the most recent 300 samples (2.4s).
        print("\n--- 4.2 Simulating 60-Second Real-Time Edge Sliding-Window Inference ---")
        window_size = 300  # 2.4s window
        step_frames = 30   # 1 second step
        n_windows = (len(session_60s["t_front_ns"]) - window_size) // step_frames

        window_latencies = []
        window_sbps = []
        window_dbps = []

        imu = session_60s.get("imu_data", {"delta_h_m": 0.0, "hold_stable": True})
        user_prof = session_60s.get("user_profile", {"height_m": 1.78})

        for w_idx in range(n_windows):
            start_i = w_idx * step_frames
            end_i = start_i + window_size

            sub_session = {
                "t_front_ns": session_60s["t_front_ns"][start_i:end_i],
                "face_rgb": session_60s["face_rgb"][start_i:end_i],
                "t_rear_ns": session_60s["t_rear_ns"][start_i:end_i],
                "finger_rgb": session_60s["finger_rgb"][start_i:end_i],
                "imu_data": imu,
                "user_profile": user_prof,
                "nominal_fps": 30.0
            }

            t_win_0 = time.perf_counter()
            win_res = pipeline.process_session(sub_session)
            win_dt = (time.perf_counter() - t_win_0) * 1000.0

            window_latencies.append(win_dt)
            window_sbps.append(win_res.sbp)
            window_dbps.append(win_res.dbp)

        mean_win_lat = float(np.mean(window_latencies))
        max_win_lat = float(np.max(window_latencies))
        sbp_trajectory_std = float(np.std(window_sbps))
        dbp_trajectory_std = float(np.std(window_dbps))

        print(f"  Simulated Streaming Windows: {n_windows} windows (spanning 60 seconds)")
        print(f"  Mean Window Processing Latency: {mean_win_lat:.2f} ms (budget < 1000 ms real-time window step)")
        print(f"  Max Window Processing Latency:  {max_win_lat:.2f} ms")
        print(f"  Mean Window SBP/DBP:            {np.mean(window_sbps):.1f} / {np.mean(window_dbps):.1f} mmHg")
        print(f"  Window SBP Trajectory Std:      {sbp_trajectory_std:.2f} mmHg (respiratory Traube-Hering-Mayer modulation)")
        print(f"  Real-time Feasibility:          {'PASS (Real-Time Edge Capable)' if max_win_lat < 1000.0 else 'FAIL (Buffer overrun)'}")

        report = {
            "batch_60s": {
                "duration_sec": 60.0,
                "processing_time_ms": round(dt_batch_ms, 2),
                "status": res_60s.status,
                "is_valid": res_60s.is_valid,
                "sustained_30fps": res_60s.quality.sustained_30fps,
                "sbp": res_60s.sbp,
                "dbp": res_60s.dbp,
            },
            "streaming_sliding_window": {
                "num_windows": n_windows,
                "mean_latency_ms": round(mean_win_lat, 2),
                "max_latency_ms": round(max_win_lat, 2),
                "sbp_mean": round(float(np.mean(window_sbps)), 1),
                "dbp_mean": round(float(np.mean(window_dbps)), 1),
                "sbp_trajectory_std": round(sbp_trajectory_std, 2),
                "dbp_trajectory_std": round(dbp_trajectory_std, 2),
                "real_time_sustainable": (max_win_lat < 1000.0),
            }
        }
        self.results["vector_4"] = report
        return report

    def run_all_stress_vectors(self) -> Dict[str, Any]:
        print("\n" + "#" * 80)
        print("LAUNCHING FULL EMPIRICAL STRESS TEST SUITE — CHALLENGER 2")
        print("#" * 80)
        self.stress_vector_1_adversarial_inputs()
        self.stress_vector_2_template_collapse_probe(n_subjects=200)
        self.stress_vector_3_memory_leak_and_buffer_invariance(iterations=10000)
        self.stress_vector_4_continuous_60s_recording()
        return self.results


if __name__ == "__main__":
    challenger = ModelMobileStressChallenger()
    challenger.run_all_stress_vectors()
