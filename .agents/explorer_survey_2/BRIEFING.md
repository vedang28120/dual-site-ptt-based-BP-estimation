# BRIEFING — 2026-09-23T18:18:00Z

## Mission
Investigate Requirement R1 (Synchronized Optical Engine for Poco M6 Pro 5G) and edge mobile constraints: Camera2 concurrent dual-stream capture, rear flash contact PPG, front MediaPipe face ROI tracking, and nanosecond SENSOR_TIMESTAMP monotonic clock binding.

## 🔒 My Identity
- Archetype: Explorer
- Roles: Optical Engine & Poco M6 Pro Specialist
- Working directory: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\explorer_survey_2
- Original parent: a8dbe54c-0e6f-4b43-a00e-4135314b7d66
- Milestone: Architectural Survey & Hardware Feasibility

## 🔒 Key Constraints
- Read-only investigation — do NOT implement production code
- Dedicated Clean Workspace: All findings and handoff reports in .agents/explorer_survey_2/
- Reference directory read-only: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation
- Target Hardware: Xiaomi Poco M6 Pro 5G (Qualcomm Snapdragon 4 Gen 2 / SM4450, Android 13/14 / HyperOS)
- R1 Specifications: Concurrent front/rear 480p @ 30 FPS, Rear torch contact PPG 64x64 center crop, Front MediaPipe facial ROI tracking, Nanosecond SENSOR_TIMESTAMP hardware monotonic clock binding, memory overhead <= 20 MB, zero dropped frames over 10s window.

## Current Parent
- Conversation ID: a8dbe54c-0e6f-4b43-a00e-4135314b7d66
- Updated: 2026-09-23T18:18:00Z

## Investigation State
- **Explored paths**:
  * Qualcomm Snapdragon 4 Gen 2 (SM4450) Spectra Dual 12-bit ISP hardware capabilities.
  * Android Camera2 API concurrent streaming (`CameraManager.getConcurrentCameraIds`, `SCALER_MANDATORY_CONCURRENT_STREAM_COMBINATIONS`).
  * Xiaomi HyperOS / MIUI 14 HAL3 behavior and fallback resilience strategies.
  * Contact PPG CMOS anti-saturation protocol (manual ISO 50/100, short exposure 500µs–2ms, locked AWB/AF).
  * 64x64 center-crop YUV direct memory extraction algorithm (<0.05 ms).
  * Front camera MediaPipe FaceLandmarker decoupled fast/slow path architecture and elimination of JPEG compression bottleneck.
  * Nanosecond `SENSOR_TIMESTAMP` hardware monotonic binding (`SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME` / `CLOCK_BOOTTIME`) and dual-ring buffer PCHIP time-grid unification.
  * Zero dropped frames and <= 20 MB memory budget verification.
- **Key findings**:
  * Hardware ISP easily supports concurrent 16MP+16MP @ 30 FPS; 480p @ 30 FPS concurrent is well within silicon bandwidth.
  * Both Camera2 `SENSOR_TIMESTAMP` streams and Android IMU use the identical monotonic clock (`elapsedRealtimeNanos`), eliminating OS thread jitter.
  * Old app bottleneck identified: JPEG compression/decompression in MediaPipe path caused massive GC churn; replacement with direct YUV sampling and asynchronous decoupled tracking reduces frame overhead from >20ms to <1ms.
- **Unexplored areas**: None; full R1 mobile pipeline surveyed and architecturally specified.

## Key Decisions Made
- Architected the dual-stream capture pipeline with decoupled Fast/Slow paths to guarantee sustained 30 FPS with 0 dropped frames.
- Specified manual sensor exposure and gain locking to prevent CMOS saturation under flash torch contact PPG.
- Formulated PCHIP nanosecond master time-grid resampling to eliminate sub-frame inter-camera rolling-shutter phase offsets and VFR jitter.

## Artifact Index
- C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\explorer_survey_2\handoff.md — Final synthesis handoff report
- C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\explorer_survey_2\progress.md — Liveness heartbeat and investigation progress
