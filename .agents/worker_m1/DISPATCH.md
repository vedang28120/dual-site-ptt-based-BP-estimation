# Dispatch Assignment — Worker M1 (Milestone 1: Synchronized Optical Engine)

## Context
You are `worker_m1` on the Dual-Site Blood Pressure Estimation Framework project.
Working directory: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m1`
Project root: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation`
Authoritative Request: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md`
Project Index: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\PROJECT.md`

## Mandatory Reading
Read `ORIGINAL_REQUEST.md`, `PROJECT.md`, and Explorer handoffs:
- `.agents/explorer_survey_2/handoff.md` (Crucial for Camera2, anti-saturation, 64x64 crop, MediaPipe, SENSOR_TIMESTAMP PCHIP sync)

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Exclusive Write Ownership
You exclusively own:
- `src/optical/`
  - `src/optical/__init__.py`
  - `src/optical/camera_config.py`
  - `src/optical/crop_extractor.py`
  - `src/optical/face_roi_tracker.py`
  - `src/optical/time_grid_synchronizer.py`
- `mobile/android/` (Companion Kotlin/Java blueprints)
  - `mobile/android/DualCameraEngine.kt`
  - `mobile/android/RearContactProcessor.kt`
  - `mobile/android/FaceLandmarkOptimizer.kt`
- `tests/unit/test_optical.py`

DO NOT modify files outside your ownership.

## Mission & Specifications
1. Implement `camera_config.py`:
   - Android Camera2 dual-stream capture configuration for Poco M6 Pro 5G (Snapdragon 4 Gen 2 Spectra ISP).
   - Rear camera manual exposure anti-saturation protocol: `CONTROL_AE_MODE_OFF`, ISO 50 floor, 1.0 ms shutter duration, fixed infinity focus, locked AWB, torch mode.
   - Front camera 480p @ 30 FPS configuration with fixed 30 FPS range and fast-path color correction.
2. Implement `crop_extractor.py`:
   - 64x64 center crop direct memory YUV_420_888 extraction (bounds: X [288..351], Y [208..271] on 640x480 frame).
   - Zero-allocation spatial averaging for Green and Red channels, avoiding CMOS saturation ($120 \le \bar{I}_G \le 190$).
3. Implement `face_roi_tracker.py`:
   - Decoupled dual-rate face tracking: Synchronous fast-path (spatial mean on EMA-smoothed bounding boxes in < 0.8 ms) and asynchronous landmarking interface.
   - Central Forehead, Left Cheek, Right Cheek ROI definitions with Eye Aspect Ratio (EAR) blink gating and 3D head yaw/pitch bounds.
4. Implement `time_grid_synchronizer.py`:
   - Nanosecond hardware monotonic clock binding (`SENSOR_TIMESTAMP`).
   - PCHIP interpolation onto a uniform 125 Hz master time grid ($8,000,000\text{ ns}$ interval).
   - Mathematically eliminates rolling-shutter phase offset and OS thread-scheduling jitter.
5. Create companion Android Kotlin production files in `mobile/android/` per Explorer 2 blueprints.
6. Write comprehensive unit tests in `tests/unit/test_optical.py` verifying anti-saturation clamp, 64x64 crop accuracy, PCHIP temporal synchronization, and sub-millisecond jitter reduction.
7. Run the tests via `pytest tests/unit/test_optical.py`, verify all pass, document results in `handoff.md`.

## 2026-09-23T18:16:31Z
You are worker_m1 (Optical Engine Implementation Specialist).
Working directory: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m1
Read C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md, C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\PROJECT.md, and C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m1\DISPATCH.md.
Review Explorer 2 handoff in .agents/explorer_survey_2/handoff.md.
MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work.
Implement src/optical/ (camera_config.py, crop_extractor.py, face_roi_tracker.py, time_grid_synchronizer.py) and companion Android Kotlin code in mobile/android/ per specifications.
Write comprehensive unit tests in tests/unit/test_optical.py.
Run pytest tests/unit/test_optical.py, verify passing, document results in handoff.md. Notify orchestrator via send_message.
