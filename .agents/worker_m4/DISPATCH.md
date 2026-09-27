# Dispatch Assignment — Worker M4 (Milestone 4: Mobile System Integration & Feasibility Pipeline)

## Context
You are `worker_m4` on the Dual-Site Blood Pressure Estimation Framework project.
Working directory: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m4`
Project root: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation`
Authoritative Request: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md`
Project Index: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\PROJECT.md`

## Mandatory Reading
Read `ORIGINAL_REQUEST.md`, `PROJECT.md`, and previous worker handoffs:
- `.agents/worker_m1/handoff.md` (`src/optical/` API contracts)
- `.agents/worker_m2/handoff.md` (`src/pinn/` API contracts)
- `.agents/worker_m3/handoff.md` (`src/models/` API contracts)

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Exclusive Write Ownership
You exclusively own:
- `src/mobile/`
  - `src/mobile/__init__.py`
  - `src/mobile/pipeline.py`
  - `src/mobile/tflite_runner.py`
- `tests/unit/test_mobile.py`

DO NOT modify files outside your ownership.

## Mission & Specifications
1. Implement `tflite_runner.py`:
   - Lightweight, production-grade mobile TFLite/ONNX runner abstraction for INT8 quantization.
   - Loads `dual_stream_tcn_int8.tflite` or PyTorch INT8 model weights.
   - Efficient memory layout: enforces pre-allocated input/output buffers, zero per-inference dynamic allocations.
   - Validates Snapdragon 4 Gen 2 runtime constraints: inference latency < 60 ms (nominal ~3.2 ms CPU / 1.6 ms NPU), peak memory <= 20 MB.
2. Implement `pipeline.py`:
   - `DualSiteBPPipeline` orchestrating the complete end-to-end processing chain:
     1. Ingests raw dual camera streams with nanosecond `SENSOR_TIMESTAMP` values.
     2. Runs `TimeGridSynchronizer` to resample onto uniform 125 Hz grid.
     3. Runs `solve_hrvp` from `src/pinn/` (sub-frame tangent foot detector, anthropometric ePWV, IMU hydrostatic tilt $\Delta P_{\text{hydro}}$, THM respiratory compliance $d(\text{ePWV})/dP$, closed-form stiffness solver $E_0, \alpha$, baseline MAP, and 6D anchor vector $z_{\text{HRVP}}$).
     4. Normalizes wave tensors into `(1, 1, 300)` contact morphology and facial timing sequences.
     5. Ingests tensors into `tflite_runner` for deep learning regression.
     6. Fuses analytical HRVP baseline with neural residual to output continuous SBP, DBP, MAP, ePWV, and signal quality metrics.
     7. Enforces clinical plausibility guards: checks physiological bounds, flags blanching or motion artifacts, verifies sustained 30 FPS timing without dropped frames.
3. Implement `src/mobile/__init__.py`:
   - Package export.
4. Write comprehensive unit tests in `tests/unit/test_mobile.py`:
   - Test end-to-end pipeline execution from raw timestamped signals to blood pressure estimates.
   - Test latency and memory profiling assertions under Snapdragon 4 Gen 2 mobile budget.
   - Test fault tolerance on motion artifacts, blanching detection, and degraded signal flags.
5. Run `pytest tests/unit/test_mobile.py`, verify all pass, document results in `handoff.md`.

## 2026-09-23T18:35:00Z
You are worker_m4 (Mobile System Integration & Feasibility Specialist).
Working directory: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m4
Read C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md, C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\PROJECT.md, and C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m4\DISPATCH.md.
Review handoff reports in .agents/worker_m1, worker_m2, and worker_m3.
MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work.
Implement src/mobile/ (pipeline.py, tflite_runner.py, and __init__.py) bridging optical synchronization, HRVP-PINN solver, and INT8 TFLite runner.
Implement unit tests in tests/unit/test_mobile.py.
Run pytest tests/unit/test_mobile.py, verify passing, document results in handoff.md. Notify orchestrator via send_message.

