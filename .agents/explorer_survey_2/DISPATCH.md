# Dispatch Assignment — Explorer 2 (Survey: Optical Engine & Poco M6 Pro Android Pipeline)

## Context
You are Explorer 2 on the Dual-Site Blood Pressure Estimation Framework project.
Working directory: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\explorer_survey_2`
Project workspace: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation`
Authoritative Request: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md`

## Mission
1. Read `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md`.
2. Thoroughly investigate Requirement R1 and edge mobile constraints:
   - Android Camera2 API multi-camera capabilities: concurrent streaming on Xiaomi Poco M6 Pro 5G (Qualcomm Snapdragon 4 Gen 2 chipset, Android 13/14 / HyperOS). Verify `CameraManager.getConcurrentCameraIds()`, dual stream 480p @ 30 FPS across front and rear.
   - Rear sensor configuration: Flash LED active mode (`FLASH_MODE_TORCH`), contact finger photoplethysmography (PPG), auto-exposure locking (`CONTROL_AE_LOCK`), manual gain control to avoid CMOS saturation, center-crop 64x64 extraction.
   - Front sensor configuration: Facial ROI tracking via MediaPipe (or FaceMesh landmarks) for forehead and cheek regions, handling lighting and motion.
   - Nanosecond clock synchronization: Hardware monotonic timestamping (`SENSOR_TIMESTAMP` via `TotalCaptureResult` / `Image.getTimestamp()`), clock skew calculation, queue buffering, timestamp-matched pairing strategy, and eliminating thread-scheduling jitter.
   - Mobile feasibility constraints: sustained 30 FPS with 0 dropped frames, memory overhead <= 20 MB.
3. Update `progress.md` in your working directory with timestamps.
4. Synthesize all findings, architectural contracts, classes, and specifications into `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\explorer_survey_2\handoff.md`.
5. Send a concise completion message back to the orchestrator.
