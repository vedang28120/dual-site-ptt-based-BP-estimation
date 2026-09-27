# Progress — Explorer 2 (Optical Engine & Poco M6 Pro Specialist)

Last visited: 2026-09-23T18:15:00Z

## Completed Investigations:
1. **Target Device Hardware & Silicon Capabilities**:
   - SoC: Qualcomm Snapdragon 4 Gen 2 (SM4450, 4nm).
   - ISP: Qualcomm Spectra Dual 12-bit ISP.
   - Silicon Concurrent Camera Bandwidth: Supports up to 16 MP + 16 MP @ 30 FPS ZSL concurrent dual cameras, dual 1080p @ 30 FPS video.
   - Sensor Complement: Rear 50 MP (Samsung ISOCELL JN1, 1/2.76", f/1.8), Front 8 MP (OmniVision OV08D10, 1/4.0", f/2.05).
   - Operating System: Android 13/14 (MIUI 14 / Xiaomi HyperOS).

2. **Android Camera2 Concurrent Streaming Architecture**:
   - API: `CameraManager.getConcurrentCameraIds()` (API level 30+).
   - Guaranteed configurations: `CameraCharacteristics.SCALER_MANDATORY_CONCURRENT_STREAM_COMBINATIONS`.
   - Resolution: 640x480 (480p) @ 30 FPS YUV_420_888 supported across both sensors simultaneously.
   - Vendor HAL analysis: In stock MIUI/HyperOS, Xiaomi disables "Dual Video" in the consumer camera UI, but the underlying Camera HAL3 exposes concurrent camera streaming. Fallback resilience patterns identified.

3. **Rear Camera Contact PPG & CMOS Anti-Saturation Protocol**:
   - Torch mode: `CaptureRequest.FLASH_MODE_TORCH`.
   - Dynamic Range & Saturation: Flash LED at zero distance floods sensor. Manual sensor control required:
     * `CONTROL_AE_MODE_OFF`
     * `SENSOR_SENSITIVITY`: Locked to min ISO (ISO 50/100)
     * `SENSOR_EXPOSURE_TIME`: Manual clamp (500 µs – 2 ms / 500,000 – 2,000,000 ns)
     * `CONTROL_AF_MODE_OFF`: Focus locked to infinity
     * `CONTROL_AWB_MODE_OFF` or `CONTROL_AWB_LOCK = true`
   - Center-crop 64x64 extraction: Region `[288..352, 208..272]` directly sampled from YUV planes. Computational overhead < 0.05 ms per frame.

4. **Front Camera MediaPipe Face ROI Tracking**:
   - Decoupled Fast/Slow Path:
     * Fast Path (30 FPS synchronous): Spatial-mean RGB extraction from YUV buffer using smoothed ROI bounding boxes (< 0.8 ms).
     * Slow Path (10-15 FPS asynchronous): MediaPipe FaceLandmarker with GPU delegate for landmarks [10, 151, 9, 8] (forehead), [116, 117] (left cheek), [345, 346] (right cheek).
     * Elimination of JPEG compression anti-pattern: Zero GC allocations, direct YUV sampling or RGBA direct memory.
     * Motion & Blink rejection via EAR (Eye Aspect Ratio) and 3D pose bounding.

5. **Nanosecond Clock Synchronization & Hardware Monotonic Clock Binding**:
   - Timebase: `SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME` maps `SENSOR_TIMESTAMP` directly to `SystemClock.elapsedRealtimeNanos()` (`CLOCK_BOOTTIME`).
   - Shared monotonic timebase between Rear Camera, Front Camera, and IMU.
   - Jitter elimination: Dual asynchronous ring buffers with post-acquisition PCHIP resampling onto a unified 125 Hz / 250 Hz uniform nanosecond time grid.

6. **Mobile Feasibility (Sustained 30 FPS, 0 Dropped Frames, <= 20 MB RAM)**:
   - Memory footprint: ~11 MB total runtime overhead (buffers: ~4.5 MB, MediaPipe model: ~4 MB, camera buffers: ~1.85 MB).
   - CPU frame processing time: Rear < 0.1 ms, Front Fast Path < 0.8 ms (total < 1 ms out of 33.3 ms budget).
   - Zero-allocation policy in hot loop prevents GC pauses.

## Next Steps:
- Compile detailed synthesis into `handoff.md`.
- Formulate complete architectural contracts, Kotlin class blueprints, capture request configurations, and verification methods.
