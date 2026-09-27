"""
Comprehensive Unit Tests for Milestone 1: Synchronized Optical Engine.

Tests:
1. Camera configuration, manual exposure anti-saturation, and dynamic closed-loop safety guard.
2. 64x64 center crop direct memory extraction, ITU-R BT.601 conversion, and saturation metrics.
3. Decoupled face ROI tracker, EMA bounding box smoothing, EAR blink gating, and head pose bounds.
4. Master Time-Grid PCHIP Resampling, nanosecond clock binding, phase offset compensation, and jitter elimination.
"""

import time
import numpy as np
import pytest

from src.optical.camera_config import (
    CameraId,
    TimestampSource,
    AEControlMode,
    CameraStreamConfig,
    ManualExposureConfig,
    FrontCameraConfig,
    DualCameraHardwareProfile,
    DynamicExposureController,
    validate_hardware_profile,
    create_rear_camera_capture_params,
    create_front_camera_capture_params,
)
from src.optical.crop_extractor import (
    CenterCropExtractor,
    CropMetrics,
    direct_yuv420_to_rgb,
    extract_64x64_center_crop,
    compute_perfusion_index,
)
from src.optical.face_roi_tracker import (
    BoundingBox,
    FaceRoiBoundingBoxes,
    HeadPoseAngles,
    FaceRoiTracker,
    compute_eye_aspect_ratio,
    estimate_head_pose_angles,
    FOREHEAD_LANDMARKS,
    LEFT_CHEEK_LANDMARKS,
    RIGHT_CHEEK_LANDMARKS,
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
)
from src.optical.time_grid_synchronizer import (
    TimeGridSynchronizer,
    compute_timestamp_jitter,
)


# ==============================================================================
# Suite 1: Camera Configuration & Anti-Saturation Protocol
# ==============================================================================

class TestCameraConfig:
    def test_dual_camera_hardware_profile(self):
        profile = DualCameraHardwareProfile()
        validation = validate_hardware_profile(profile)
        assert validation["valid"] is True
        assert validation["checks"]["concurrent_supported"] is True
        assert validation["checks"]["timestamp_realtime"] is True
        assert validation["checks"]["contains_rear_and_front"] is True
        assert validation["checks"]["isp_headroom_sufficient"] is True

        front_stream = CameraStreamConfig(camera_id=CameraId.FRONT, width=640, height=480, fps_range=(30, 30))
        rear_stream = CameraStreamConfig(camera_id=CameraId.REAR, width=640, height=480, fps_range=(30, 30))
        headroom = profile.compute_bandwidth_headroom(front_stream, rear_stream)
        # Dual 480p @ 30 FPS uses ~18.4 MP/sec combined out of 32 MP/sec (>40% headroom)
        assert headroom > 40.0

    def test_manual_exposure_defaults_and_clamping(self):
        config = ManualExposureConfig()
        assert config.ae_mode == AEControlMode.OFF
        assert config.iso == 50
        assert config.exposure_time_ns == 1_000_000  # 1.0 ms
        assert config.lens_focus_distance == 0.0      # Fixed infinity

        # Test safety clamping
        assert config.clamp_exposure(50_000) == config.min_exposure_ns
        assert config.clamp_exposure(20_000_000) == config.max_exposure_ns
        assert config.clamp_exposure(1_500_000) == 1_500_000

    def test_dynamic_exposure_controller_saturation_reduction(self):
        controller = DynamicExposureController(
            config=ManualExposureConfig(exposure_time_ns=2_000_000),
            saturation_threshold=220.0,
            adjustment_factor=0.20,
        )
        # Saturated frame (mean_green = 240 > 220)
        new_exp, is_locked = controller.update(mean_green=240.0)
        assert new_exp == int(2_000_000 * 0.8)
        assert not is_locked

    def test_dynamic_exposure_controller_underexposure_boost(self):
        controller = DynamicExposureController(
            config=ManualExposureConfig(exposure_time_ns=500_000),
            underexposure_threshold=80.0,
            adjustment_factor=0.20,
        )
        # Underexposed frame (mean_green = 60 < 80)
        new_exp, is_locked = controller.update(mean_green=60.0)
        assert new_exp == int(500_000 * 1.2)
        assert not is_locked

    def test_dynamic_exposure_controller_optimal_lock(self):
        controller = DynamicExposureController(
            config=ManualExposureConfig(exposure_time_ns=1_000_000),
            target_min=120.0,
            target_max=190.0,
        )
        # Frame inside optimal transillumination regime (mean_green = 150)
        new_exp, is_locked = controller.update(mean_green=150.0)
        assert is_locked is True
        assert new_exp == 1_000_000

        # Subsequent updates should not change locked exposure
        subsequent_exp, locked = controller.update(mean_green=245.0)
        assert locked is True
        assert subsequent_exp == 1_000_000

    def test_dynamic_exposure_controller_max_calibration_lock(self):
        controller = DynamicExposureController(max_calibration_frames=5)
        for _ in range(5):
            _, is_locked = controller.update(mean_green=230.0)
        assert is_locked is True

    def test_capture_params_generation(self):
        manual_cfg = ManualExposureConfig()
        rear_params = create_rear_camera_capture_params(manual_cfg)
        assert rear_params["CONTROL_AE_MODE"] == "CONTROL_AE_MODE_OFF"
        assert rear_params["SENSOR_SENSITIVITY"] == 50
        assert rear_params["SENSOR_EXPOSURE_TIME"] == 1_000_000
        assert rear_params["FLASH_MODE"] == "FLASH_MODE_TORCH"
        assert rear_params["CONTROL_AWB_LOCK"] is True

        front_cfg = FrontCameraConfig()
        front_params = create_front_camera_capture_params(front_cfg)
        assert front_params["CONTROL_AE_MODE"] == "CONTROL_AE_MODE_ON"
        assert front_params["CONTROL_AE_TARGET_FPS_RANGE"] == [30, 30]


# ==============================================================================
# Suite 2: 64x64 Center Crop & Contact PPG Extraction
# ==============================================================================

class TestCropExtractor:
    def test_crop_bounds_and_dimensions(self):
        extractor = CenterCropExtractor(frame_width=640, frame_height=480, crop_size=64)
        x0, x1, y0, y1 = extractor.get_crop_bounds()
        assert x0 == 288
        assert x1 == 352
        assert y0 == 208
        assert y1 == 272
        assert (x1 - x0) == 64
        assert (y1 - y0) == 64
        assert extractor.num_pixels == 4096

    def test_direct_rgb_extraction_accuracy(self):
        extractor = CenterCropExtractor()
        # Synthetic 640x480 frame with constant color
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [180, 150, 40]  # Red, Green, Blue

        metrics = extractor.extract_from_rgb(frame)
        assert pytest.approx(metrics.mean_r, rel=1e-3) == 180.0
        assert pytest.approx(metrics.mean_g, rel=1e-3) == 150.0
        assert pytest.approx(metrics.mean_b, rel=1e-3) == 40.0
        assert metrics.saturated_pixel_count == 0
        assert metrics.saturated_fraction == 0.0
        assert metrics.in_optimal_regime is True
        assert metrics.is_saturated is False
        assert metrics.is_underexposed is False

    def test_saturation_detection(self):
        extractor = CenterCropExtractor()
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)

        metrics = extractor.extract_from_rgb(frame)
        assert metrics.is_saturated is True
        assert metrics.in_optimal_regime is False
        assert metrics.saturated_fraction == 1.0

    def test_underexposure_detection(self):
        extractor = CenterCropExtractor()
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)

        metrics = extractor.extract_from_rgb(frame)
        assert metrics.is_underexposed is True
        assert metrics.in_optimal_regime is False

    def test_direct_yuv420_conversion(self):
        # Test BT.601 conversion with pure neutral gray (Y=128, U=0, V=0 relative to 128)
        r, g, b = direct_yuv420_to_rgb(128, 0, 0)
        assert pytest.approx(r, abs=1.0) == 128.0
        assert pytest.approx(g, abs=1.0) == 128.0
        assert pytest.approx(b, abs=1.0) == 128.0

    def test_yuv_plane_extraction_simulation(self):
        extractor = CenterCropExtractor()
        width, height = 640, 480
        y_stride = 640
        uv_stride = 320

        # Uniform gray transillumination: Y=150, U=128 (centered 0), V=148 (centered +20)
        y_plane = bytearray([150] * (width * height))
        u_plane = bytearray([128] * ((width // 2) * (height // 2)))
        v_plane = bytearray([148] * ((width // 2) * (height // 2)))

        metrics = extractor.extract_from_yuv_planes(
            y_plane=bytes(y_plane),
            u_plane=bytes(u_plane),
            v_plane=bytes(v_plane),
            y_row_stride=y_stride,
            uv_row_stride=uv_stride,
            uv_pixel_stride=1,
        )

        expected_r = min(255.0, 150 + 1.402 * 20)
        expected_g = min(255.0, 150 - 0.714136 * 20)
        expected_b = 150.0

        assert pytest.approx(metrics.mean_r, abs=1.0) == expected_r
        assert pytest.approx(metrics.mean_g, abs=1.0) == expected_g
        assert pytest.approx(metrics.mean_b, abs=1.0) == expected_b
        assert metrics.pixel_count == 4096

    def test_extract_crop_patch_shape(self):
        extractor = CenterCropExtractor()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        patch = extractor.extract_crop_patch(frame)
        assert patch.shape == (64, 64, 3)

    def test_perfusion_index(self):
        # 1.5% AC amplitude on DC of 150 -> PI = 1.0%
        pi = compute_perfusion_index(ac_amplitude=1.5, dc_baseline=150.0)
        assert pytest.approx(pi, rel=1e-3) == 1.0


# ==============================================================================
# Suite 3: Decoupled Face ROI Tracker & Gating
# ==============================================================================

class TestFaceRoiTracker:
    def test_bounding_box_smooth_and_coords(self):
        b0 = BoundingBox(0.40, 0.20, 0.60, 0.30)
        b_target = BoundingBox(0.50, 0.30, 0.70, 0.40)
        alpha = 0.20
        b_smooth = b0.smooth_with(b_target, alpha)

        expected_x_min = 0.40 * 0.80 + 0.50 * 0.20  # 0.42
        assert pytest.approx(b_smooth.x_min, rel=1e-4) == expected_x_min

        x0, x1, y0, y1 = b_smooth.to_pixel_coords(width=640, height=480)
        assert 0 <= x0 < x1 <= 640
        assert 0 <= y0 < y1 <= 480

    def test_eye_aspect_ratio_open_vs_closed(self):
        # Open eye geometry
        open_eye = np.array([
            [0.10, 0.20],  # p1
            [0.15, 0.18],  # p2
            [0.25, 0.18],  # p3
            [0.30, 0.20],  # p4
            [0.25, 0.22],  # p5
            [0.15, 0.22],  # p6
        ], dtype=np.float32)
        ear_open = compute_eye_aspect_ratio(open_eye)
        assert ear_open > 0.15  # Typically ~0.20 - 0.35

        # Closed eye geometry (p2 near p6, p3 near p5)
        closed_eye = np.array([
            [0.10, 0.200],  # p1
            [0.15, 0.199],  # p2
            [0.25, 0.199],  # p3
            [0.30, 0.200],  # p4
            [0.25, 0.201],  # p5
            [0.15, 0.201],  # p6
        ], dtype=np.float32)
        ear_closed = compute_eye_aspect_ratio(closed_eye)
        assert ear_closed < 0.05
        assert ear_closed < 0.20  # Flags blink

    def test_head_pose_angles_and_bounds(self):
        # Construct synthetic 468 landmarks with centered face
        landmarks = np.zeros((468, 3), dtype=np.float32)
        landmarks[1] = [0.50, 0.50, 0.0]     # Nose
        landmarks[10] = [0.50, 0.20, 0.0]    # Forehead
        landmarks[152] = [0.50, 0.80, 0.0]   # Chin
        landmarks[33] = [0.40, 0.40, 0.0]    # Left eye
        landmarks[263] = [0.60, 0.40, 0.0]   # Right eye

        pose = estimate_head_pose_angles(landmarks)
        assert abs(pose.yaw_deg) < 5.0
        assert abs(pose.pitch_deg) < 5.0
        assert pose.is_within_bounds is True

        # Test extreme yaw rotation
        landmarks_turned = landmarks.copy()
        landmarks_turned[1] = [0.58, 0.50, 0.0]  # Nose shifted strongly right
        pose_turned = estimate_head_pose_angles(landmarks_turned)
        assert abs(pose_turned.yaw_deg) > 12.0
        assert pose_turned.is_within_bounds is False

    def test_fast_path_latency_and_extraction(self):
        tracker = FaceRoiTracker()
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

        # Synchronous fast-path execution
        metrics = tracker.process_frame_fast_path(frame)
        assert metrics.forehead_rgb.shape == (3,)
        assert metrics.left_cheek_rgb.shape == (3,)
        assert metrics.right_cheek_rgb.shape == (3,)
        assert metrics.combined_rppg.shape == (3,)
        # Synchronous frame latency budget constraint: < 0.8 ms
        assert metrics.latency_ms < 5.0  # Even in un-optimized python, < 5 ms; on A78 native < 0.8 ms


# ==============================================================================
# Suite 4: Master Time-Grid PCHIP Resampling & Synchronization
# ==============================================================================

class TestTimeGridSynchronizer:
    def test_nanosecond_pchip_synchronization(self):
        synchronizer = TimeGridSynchronizer(target_fs_hz=125.0)
        assert synchronizer.dt_ns == 8_000_000  # 8 ms

        # Simulate 3 seconds of asynchronous front and rear acquisition at ~30 FPS
        # Rear stream starts 14.2 ms after front stream (inter-camera phase offset)
        t_front = np.arange(0, 3_000_000_000, 33_333_333, dtype=np.int64)  # ~30 FPS
        # Add realistic VFR jitter (+/- 2 ms)
        rng = np.random.RandomState(42)
        front_jitter = rng.randint(-2_000_000, 2_000_000, size=len(t_front))
        front_jitter[0] = 0
        front_jitter[-1] = 0
        t_front += front_jitter

        t_rear = np.arange(14_200_000, 3_000_000_000, 33_333_333, dtype=np.int64)
        rear_jitter = rng.randint(-2_000_000, 2_000_000, size=len(t_rear))
        rear_jitter[0] = 0
        rear_jitter[-1] = 0
        t_rear += rear_jitter

        # Synthetic cardiac signals (1.2 Hz heartbeat)
        t_f_sec = t_front / 1e9
        face_signal = np.sin(2 * np.pi * 1.2 * t_f_sec)[:, np.newaxis].repeat(3, axis=1)

        t_r_sec = t_rear / 1e9
        # Rear pulse delayed by 100 ms (PTT delay)
        finger_signal = np.sin(2 * np.pi * 1.2 * (t_r_sec - 0.100))[:, np.newaxis].repeat(3, axis=1)

        result = synchronizer.synchronize_streams(
            t_front_ns=t_front,
            face_rgb=face_signal,
            t_rear_ns=t_rear,
            finger_rgb=finger_signal,
            nominal_fps=30.0,
        )

        assert result.num_samples > 300
        assert result.target_fs_hz == 125.0
        assert result.resampled_face.shape == (result.num_samples, 3)
        assert result.resampled_finger.shape == (result.num_samples, 3)

        # Verify exact equispaced master grid (0.0 ms jitter)
        dt_grid_ns = np.diff(result.master_time_ns)
        assert np.all(dt_grid_ns == 8_000_000)
        assert result.max_residual_sync_error_ms == 0.0

    def test_zero_temporal_overlap_raises_error(self):
        synchronizer = TimeGridSynchronizer()
        t_front = np.array([1_000_000_000, 1_033_000_000], dtype=np.int64)
        t_rear = np.array([2_000_000_000, 2_033_000_000], dtype=np.int64)
        face_rgb = np.zeros((2, 3), dtype=np.float32)
        finger_rgb = np.zeros((2, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Zero temporal overlap"):
            synchronizer.synchronize_streams(t_front, face_rgb, t_rear, finger_rgb)

    def test_jitter_computation(self):
        nominal_interval_ns = 33_333_333
        timestamps = np.arange(0, 10 * nominal_interval_ns, nominal_interval_ns, dtype=np.int64)
        jitter_stats = compute_timestamp_jitter(timestamps, nominal_fps=30.0)
        assert pytest.approx(jitter_stats["mean_interval_ms"], rel=1e-3) == 33.333
        assert pytest.approx(jitter_stats["std_jitter_ms"], abs=1e-3) == 0.0
        assert jitter_stats["dropped_frames"] == 0.0

        # Inject one dropped frame (gap of 70 ms)
        timestamps_dropped = timestamps.copy()
        timestamps_dropped[5:] += 40_000_000
        stats_dropped = compute_timestamp_jitter(timestamps_dropped, nominal_fps=30.0)
        assert stats_dropped["dropped_frames"] == 1.0
