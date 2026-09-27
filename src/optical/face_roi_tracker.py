"""
Decoupled Dual-Rate Face ROI Tracking and Gating for Front Camera rPPG.

Solves the MediaPipe JNI/GC bottleneck observed in prior implementations:
- Synchronous Fast Path (30 FPS, frame budget < 0.8 ms):
  Direct spatial mean sampling over Exponential Moving Average (EMA) smoothed ROIs.
- Asynchronous Slow Path (10-15 FPS off-thread):
  Processes landmarks (468 keypoints) for Forehead, Left Cheek, Right Cheek,
  evaluates Eye Aspect Ratio (EAR) blink gating (< 0.20), and 3D head yaw/pitch bounds (<= 12 degrees).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# Standard 468 MediaPipe FaceMesh canonical landmark indices
FOREHEAD_LANDMARKS = [10, 151, 9, 8, 108, 337, 336, 107]
LEFT_CHEEK_LANDMARKS = [116, 117, 118, 119, 100, 126, 209, 49]
RIGHT_CHEEK_LANDMARKS = [345, 346, 347, 348, 329, 355, 429, 279]

# Eye landmarks for Eye Aspect Ratio (EAR) computation
# [p1 (lateral corner), p2 (top-lateral), p3 (top-medial), p4 (medial corner), p5 (bottom-medial), p6 (bottom-lateral)]
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Head pose reference landmarks: nose tip (1), chin (152), forehead (10), left eye (33), right eye (263)
HEAD_POSE_LANDMARKS = {"nose": 1, "chin": 152, "forehead": 10, "left_eye": 33, "right_eye": 263}


@dataclass
class BoundingBox:
    """Normalized ROI bounding box [0.0, 1.0]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def clamp(self) -> BoundingBox:
        """Clamp coordinates within [0.0, 1.0] and ensure valid bounds."""
        x0 = max(0.0, min(1.0, self.x_min))
        x1 = max(x0 + 1e-4, min(1.0, self.x_max))
        y0 = max(0.0, min(1.0, self.y_min))
        y1 = max(y0 + 1e-4, min(1.0, self.y_max))
        return BoundingBox(x0, y0, x1, y1)

    def to_pixel_coords(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized bounding box to pixel slice coordinates [x0, x1, y0, y1]."""
        x0 = int(np.clip(np.floor(self.x_min * width), 0, width - 1))
        x1 = int(np.clip(np.ceil(self.x_max * width), x0 + 1, width))
        y0 = int(np.clip(np.floor(self.y_min * height), 0, height - 1))
        y1 = int(np.clip(np.ceil(self.y_max * height), y0 + 1, height))
        return x0, x1, y0, y1

    def smooth_with(self, target: BoundingBox, alpha: float) -> BoundingBox:
        """Apply Exponential Moving Average (EMA): B_smooth = (1 - a) * B_old + a * B_target."""
        return BoundingBox(
            x_min=(1.0 - alpha) * self.x_min + alpha * target.x_min,
            y_min=(1.0 - alpha) * self.y_min + alpha * target.y_min,
            x_max=(1.0 - alpha) * self.x_max + alpha * target.x_max,
            y_max=(1.0 - alpha) * self.y_max + alpha * target.y_max,
        ).clamp()


@dataclass
class FaceRoiBoundingBoxes:
    """Set of facial ROIs used for multi-site spatial rPPG."""
    forehead: BoundingBox
    left_cheek: BoundingBox
    right_cheek: BoundingBox

    def smooth_with(self, target: FaceRoiBoundingBoxes, alpha: float) -> FaceRoiBoundingBoxes:
        return FaceRoiBoundingBoxes(
            forehead=self.forehead.smooth_with(target.forehead, alpha),
            left_cheek=self.left_cheek.smooth_with(target.left_cheek, alpha),
            right_cheek=self.right_cheek.smooth_with(target.right_cheek, alpha),
        )


@dataclass(frozen=True)
class HeadPoseAngles:
    """Estimated head orientation angles in degrees."""
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    @property
    def is_within_bounds(self) -> bool:
        """Check if head rotation is within permissible bounds (|yaw| <= 12 deg, |pitch| <= 12 deg)."""
        return abs(self.yaw_deg) <= 12.0 and abs(self.pitch_deg) <= 12.0


@dataclass
class FaceTrackingMetrics:
    """Output metrics from synchronous fast-path frame processing."""
    forehead_rgb: np.ndarray    # Shape (3,)
    left_cheek_rgb: np.ndarray  # Shape (3,)
    right_cheek_rgb: np.ndarray # Shape (3,)
    combined_rppg: np.ndarray   # Shape (3,) weighted face rPPG (0.6 Forehead + 0.2 Left + 0.2 Right)
    ear: float
    is_blinking: bool
    head_pose: HeadPoseAngles
    is_motion_valid: bool
    latency_ms: float


def compute_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio (EAR) from 6 2D/3D landmark coordinates.
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if eye_landmarks.shape[0] < 6:
        raise ValueError("EAR computation requires at least 6 eye landmarks.")

    p1 = eye_landmarks[0, :2]
    p2 = eye_landmarks[1, :2]
    p3 = eye_landmarks[2, :2]
    p4 = eye_landmarks[3, :2]
    p5 = eye_landmarks[4, :2]
    p6 = eye_landmarks[5, :2]

    dist_vert1 = float(np.linalg.norm(p2 - p6))
    dist_vert2 = float(np.linalg.norm(p3 - p5))
    dist_horiz = float(np.linalg.norm(p1 - p4))

    if dist_horiz < 1e-6:
        return 0.0

    ear = (dist_vert1 + dist_vert2) / (2.0 * dist_horiz)
    return float(ear)


def estimate_head_pose_angles(landmarks: np.ndarray) -> HeadPoseAngles:
    """
    Fast geometric head pose estimation from canonical facial keypoints.
    landmarks: shape (N, 3) or (N, 2) normalized face mesh coordinates.
    """
    if landmarks.shape[0] < 468:
        # Default zero pose if full landmark set is not present
        return HeadPoseAngles(0.0, 0.0, 0.0)

    nose = landmarks[HEAD_POSE_LANDMARKS["nose"]]
    forehead = landmarks[HEAD_POSE_LANDMARKS["forehead"]]
    chin = landmarks[HEAD_POSE_LANDMARKS["chin"]]
    left_eye = landmarks[HEAD_POSE_LANDMARKS["left_eye"]]
    right_eye = landmarks[HEAD_POSE_LANDMARKS["right_eye"]]

    # Yaw: ratio of horizontal distance from nose to left eye vs right eye
    dist_nose_left = abs(nose[0] - left_eye[0])
    dist_nose_right = abs(nose[0] - right_eye[0])
    total_interocular = dist_nose_left + dist_nose_right + 1e-6
    asymmetry = (dist_nose_right - dist_nose_left) / total_interocular
    # Linear projection calibrated to roughly +/-45 degrees at extremes
    yaw_deg = float(np.clip(asymmetry * 60.0, -90.0, 90.0))

    # Pitch: vertical position of nose relative to forehead-chin midpoint
    face_height = abs(chin[1] - forehead[1]) + 1e-6
    midpoint_y = (chin[1] + forehead[1]) / 2.0
    pitch_offset = (nose[1] - midpoint_y) / face_height
    pitch_deg = float(np.clip(pitch_offset * 70.0, -90.0, 90.0))

    # Roll: angle of interocular vector relative to horizontal
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll_rad = np.arctan2(dy, dx + 1e-6)
    roll_deg = float(np.degrees(roll_rad))

    return HeadPoseAngles(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)


def compute_bounding_box_from_landmarks(
    landmarks: np.ndarray,
    indices: List[int],
    padding_x: float = 0.01,
    padding_y: float = 0.01,
) -> BoundingBox:
    """Extract tight bounding box enclosing specified landmark indices with margin."""
    pts = landmarks[indices, :2]
    x_min = float(np.min(pts[:, 0])) - padding_x
    x_max = float(np.max(pts[:, 0])) + padding_x
    y_min = float(np.min(pts[:, 1])) - padding_y
    y_max = float(np.max(pts[:, 1])) + padding_y
    return BoundingBox(x_min, y_min, x_max, y_max).clamp()


class FaceRoiTracker:
    """
    Decoupled face ROI tracker maintaining synchronous fast-path execution (< 0.8 ms)
    and asynchronous landmark updates.
    """
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        ema_alpha: float = 0.15,
        ear_blink_threshold: float = 0.20,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ema_alpha = ema_alpha
        self.ear_blink_threshold = ear_blink_threshold

        # Default canonical centered face ROIs in normalized coordinates
        self.current_rois = FaceRoiBoundingBoxes(
            forehead=BoundingBox(0.40, 0.15, 0.60, 0.28),
            left_cheek=BoundingBox(0.28, 0.45, 0.42, 0.60),
            right_cheek=BoundingBox(0.58, 0.45, 0.72, 0.60),
        )

        self.latest_target_rois = self.current_rois
        self.latest_ear = 0.30
        self.is_blinking = False
        self.latest_head_pose = HeadPoseAngles(0.0, 0.0, 0.0)
        self.is_motion_valid = True

    def update_landmarks_async(self, landmarks_468: np.ndarray) -> None:
        """
        Asynchronous slow-path update callback (10-15 FPS off-thread).
        Takes (468, 2 or 3) normalized landmarks array from MediaPipe FaceLandmarker.
        """
        if landmarks_468.shape[0] < 468:
            return

        # 1. Compute target ROI bounding boxes
        target_forehead = compute_bounding_box_from_landmarks(
            landmarks_468, FOREHEAD_LANDMARKS, padding_x=0.015, padding_y=0.015
        )
        target_left_cheek = compute_bounding_box_from_landmarks(
            landmarks_468, LEFT_CHEEK_LANDMARKS, padding_x=0.015, padding_y=0.015
        )
        target_right_cheek = compute_bounding_box_from_landmarks(
            landmarks_468, RIGHT_CHEEK_LANDMARKS, padding_x=0.015, padding_y=0.015
        )

        self.latest_target_rois = FaceRoiBoundingBoxes(
            forehead=target_forehead,
            left_cheek=target_left_cheek,
            right_cheek=target_right_cheek,
        )

        # 2. Compute Eye Aspect Ratio (EAR) for blink gating
        ear_left = compute_eye_aspect_ratio(landmarks_468[LEFT_EYE_LANDMARKS])
        ear_right = compute_eye_aspect_ratio(landmarks_468[RIGHT_EYE_LANDMARKS])
        self.latest_ear = float((ear_left + ear_right) / 2.0)
        self.is_blinking = self.latest_ear < self.ear_blink_threshold

        # 3. Estimate Head Pose Angles and stability
        self.latest_head_pose = estimate_head_pose_angles(landmarks_468)
        self.is_motion_valid = (not self.is_blinking) and self.latest_head_pose.is_within_bounds

    def process_frame_fast_path(self, rgb_frame: np.ndarray) -> FaceTrackingMetrics:
        """
        Synchronous fast-path extraction executed on camera callback thread.
        Budget: < 0.8 ms per frame.
        """
        t0 = time.perf_counter()

        # 1. Update EMA-smoothed bounding boxes
        self.current_rois = self.current_rois.smooth_with(
            self.latest_target_rois, self.ema_alpha
        )

        # 2. Direct spatial mean extraction over ROI slices
        h, w, _ = rgb_frame.shape
        forehead_box = self.current_rois.forehead.to_pixel_coords(w, h)
        left_box = self.current_rois.left_cheek.to_pixel_coords(w, h)
        right_box = self.current_rois.right_cheek.to_pixel_coords(w, h)

        # Forehead slice
        f_patch = rgb_frame[forehead_box[2]:forehead_box[3], forehead_box[0]:forehead_box[1], :]
        forehead_rgb = np.mean(f_patch, axis=(0, 1)).astype(np.float32)

        # Left cheek slice
        l_patch = rgb_frame[left_box[2]:left_box[3], left_box[0]:left_box[1], :]
        left_cheek_rgb = np.mean(l_patch, axis=(0, 1)).astype(np.float32)

        # Right cheek slice
        r_patch = rgb_frame[right_box[2]:right_box[3], right_box[0]:right_box[1], :]
        right_cheek_rgb = np.mean(r_patch, axis=(0, 1)).astype(np.float32)

        # Weighted combination: Forehead carries highest pulsatile SNR (60%), cheeks (20% each)
        combined_rppg = (
            0.60 * forehead_rgb + 0.20 * left_cheek_rgb + 0.20 * right_cheek_rgb
        ).astype(np.float32)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return FaceTrackingMetrics(
            forehead_rgb=forehead_rgb,
            left_cheek_rgb=left_cheek_rgb,
            right_cheek_rgb=right_cheek_rgb,
            combined_rppg=combined_rppg,
            ear=self.latest_ear,
            is_blinking=self.is_blinking,
            head_pose=self.latest_head_pose,
            is_motion_valid=self.is_motion_valid,
            latency_ms=latency_ms,
        )
