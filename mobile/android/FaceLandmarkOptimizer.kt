package com.dualsite.bp.optical

import android.media.Image
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*

/**
 * Decoupled Dual-Rate Face ROI Tracker and Landmark Optimizer.
 *
 * Eliminates the legacy Bitmap JPEG compression / decompression bottleneck:
 * - Fast Path (Synchronous @ 30 FPS, budget < 0.8 ms):
 *   Computes spatial mean RGB across EMA-smoothed bounding boxes directly on camera thread.
 * - Slow Path (Asynchronous @ 10-15 FPS off-thread):
 *   Runs MediaPipe FaceLandmarker, computes EAR blink gating, head yaw/pitch bounds,
 *   and atomically updates ROI targets.
 */
class FaceLandmarkOptimizer(
    private val frameWidth: Int = 640,
    private val frameHeight: Int = 480,
    private val emaAlpha: Float = 0.15f,
    private val earThreshold: Float = 0.20f
) {
    data class NormalizedRect(
        val xMin: Float,
        val yMin: Float,
        val xMax: Float,
        val yMax: Float
    ) {
        fun smoothWith(target: NormalizedRect, alpha: Float): NormalizedRect {
            return NormalizedRect(
                xMin = (1f - alpha) * xMin + alpha * target.xMin,
                yMin = (1f - alpha) * yMin + alpha * target.yMin,
                xMax = (1f - alpha) * xMax + alpha * target.xMax,
                yMax = (1f - alpha) * yMax + alpha * target.yMax
            )
        }
    }

    data class FaceRoiTargets(
        val forehead: NormalizedRect,
        val leftCheek: NormalizedRect,
        val rightCheek: NormalizedRect
    )

    data class FaceExtractionResult(
        val foreheadRgb: FloatArray,
        val leftCheekRgb: FloatArray,
        val rightCheekRgb: FloatArray,
        val combinedRppg: FloatArray,
        val ear: Float,
        val isBlinking: Boolean,
        val isMotionValid: Boolean,
        val latencyMs: Float
    )

    // Canonical MediaPipe landmark indices
    companion object {
        val FOREHEAD_LANDMARKS = intArrayOf(10, 151, 9, 8, 108, 337, 336, 107)
        val LEFT_CHEEK_LANDMARKS = intArrayOf(116, 117, 118, 119, 100, 126, 209, 49)
        val RIGHT_CHEEK_LANDMARKS = intArrayOf(345, 346, 347, 348, 329, 355, 429, 279)
        val LEFT_EYE_LANDMARKS = intArrayOf(33, 160, 158, 133, 153, 144)
        val RIGHT_EYE_LANDMARKS = intArrayOf(362, 385, 387, 263, 373, 380)
    }

    // Default centered ROIs in normalized coordinates
    @Volatile
    private var targetRois = FaceRoiTargets(
        forehead = NormalizedRect(0.40f, 0.15f, 0.60f, 0.28f),
        leftCheek = NormalizedRect(0.28f, 0.45f, 0.42f, 0.60f),
        rightCheek = NormalizedRect(0.58f, 0.45f, 0.72f, 0.60f)
    )

    private var currentRois = targetRois

    @Volatile
    private var latestEar: Float = 0.30f
    @Volatile
    private var isBlinking: Boolean = false
    @Volatile
    private var isMotionValid: Boolean = true

    private val landmarkExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    /**
     * Fast-path synchronous ROI extraction on camera callback thread.
     * Takes < 0.8 ms, avoiding frame drops.
     */
    fun processFastPath(image: Image): FaceExtractionResult {
        val t0 = System.nanoTime()

        // 1. Update EMA-smoothed bounding boxes
        val targets = targetRois
        currentRois = FaceRoiTargets(
            forehead = currentRois.forehead.smoothWith(targets.forehead, emaAlpha),
            leftCheek = currentRois.leftCheek.smoothWith(targets.leftCheek, emaAlpha),
            rightCheek = currentRois.rightCheek.smoothWith(targets.rightCheek, emaAlpha)
        )

        // 2. Direct spatial average extraction from YUV planes
        val foreheadRgb = extractRectAverage(image, currentRois.forehead)
        val leftRgb = extractRectAverage(image, currentRois.leftCheek)
        val rightRgb = extractRectAverage(image, currentRois.rightCheek)

        // Weighted combination: Forehead (60%), Cheeks (20% each)
        val combined = FloatArray(3) { i ->
            0.60f * foreheadRgb[i] + 0.20f * leftRgb[i] + 0.20f * rightRgb[i]
        }

        val latencyMs = (System.nanoTime() - t0) / 1_000_000.0f

        return FaceExtractionResult(
            foreheadRgb = foreheadRgb,
            leftCheekRgb = leftRgb,
            rightCheekRgb = rightRgb,
            combinedRppg = combined,
            ear = latestEar,
            isBlinking = isBlinking,
            isMotionValid = isMotionValid,
            latencyMs = latencyMs
        )
    }

    /**
     * Slow-path asynchronous landmark callback (10-15 FPS off-thread).
     * landmarks: Flattened normalized coordinates (x, y, z) for 468 points.
     */
    fun onLandmarksComputed(landmarks: FloatArray) {
        if (landmarks.size < 468 * 3) return

        // 1. Compute bounding boxes
        val newForehead = computeBoundingBox(landmarks, FOREHEAD_LANDMARKS, 0.015f)
        val newLeftCheek = computeBoundingBox(landmarks, LEFT_CHEEK_LANDMARKS, 0.015f)
        val newRightCheek = computeBoundingBox(landmarks, RIGHT_CHEEK_LANDMARKS, 0.015f)

        targetRois = FaceRoiTargets(newForehead, newLeftCheek, newRightCheek)

        // 2. Eye Aspect Ratio (EAR) blink gating
        val leftEar = computeEyeAspectRatio(landmarks, LEFT_EYE_LANDMARKS)
        val rightEar = computeEyeAspectRatio(landmarks, RIGHT_EYE_LANDMARKS)
        latestEar = (leftEar + rightEar) / 2.0f
        isBlinking = latestEar < earThreshold

        // 3. Head Pose stability bounds (|yaw| <= 12 deg, |pitch| <= 12 deg)
        val (yawDeg, pitchDeg) = estimateYawPitch(landmarks)
        isMotionValid = !isBlinking && abs(yawDeg) <= 12f && abs(pitchDeg) <= 12f
    }

    private fun extractRectAverage(image: Image, rect: NormalizedRect): FloatArray {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val x0 = (rect.xMin * frameWidth).toInt().coerceIn(0, frameWidth - 1)
        val x1 = (rect.xMax * frameWidth).toInt().coerceIn(x0 + 1, frameWidth)
        val y0 = (rect.yMin * frameHeight).toInt().coerceIn(0, frameHeight - 1)
        val y1 = (rect.yMax * frameHeight).toInt().coerceIn(y0 + 1, frameHeight)

        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var count = 0

        // Subsample step to guarantee < 0.2 ms per ROI
        val step = 2

        for (y in y0 until y1 step step) {
            val yOffset = y * yRowStride
            val uvOffset = (y shr 1) * uvRowStride

            for (x in x0 until x1 step step) {
                val yIndex = yOffset + x
                val uvIndex = uvOffset + (x shr 1) * uvPixelStride

                val yVal = yBuffer.get(yIndex).toInt() and 0xFF
                val uVal = (uBuffer.get(uvIndex).toInt() and 0xFF) - 128
                val vVal = (vBuffer.get(uvIndex).toInt() and 0xFF) - 128

                val r = (yVal + 1.402f * vVal).coerceIn(0f, 255f)
                val g = (yVal - 0.344136f * uVal - 0.714136f * vVal).coerceIn(0f, 255f)
                val b = (yVal + 1.772f * uVal).coerceIn(0f, 255f)

                sumR += r
                sumG += g
                sumB += b
                count++
            }
        }

        val total = max(1, count)
        return floatArrayOf(
            (sumR / total).toFloat(),
            (sumG / total).toFloat(),
            (sumB / total).toFloat()
        )
    }

    private fun computeBoundingBox(landmarks: FloatArray, indices: IntArray, padding: Float): NormalizedRect {
        var minX = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE

        for (idx in indices) {
            val x = landmarks[idx * 3]
            val y = landmarks[idx * 3 + 1]
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }

        return NormalizedRect(
            (minX - padding).coerceIn(0f, 1f),
            (minY - padding).coerceIn(0f, 1f),
            (maxX + padding).coerceIn(0f, 1f),
            (maxY + padding).coerceIn(0f, 1f)
        )
    }

    private fun computeEyeAspectRatio(landmarks: FloatArray, eyeIndices: IntArray): Float {
        // Points: p1 (0), p2 (1), p3 (2), p4 (3), p5 (4), p6 (5)
        fun dist(i: Int, j: Int): Float {
            val idxI = eyeIndices[i] * 3
            val idxJ = eyeIndices[j] * 3
            val dx = landmarks[idxI] - landmarks[idxJ]
            val dy = landmarks[idxI + 1] - landmarks[idxJ + 1]
            return sqrt(dx * dx + dy * dy)
        }

        val p2p6 = dist(1, 5)
        val p3p5 = dist(2, 4)
        val p1p4 = dist(0, 3)

        return if (p1p4 > 1e-5f) (p2p6 + p3p5) / (2.0f * p1p4) else 0f
    }

    private fun estimateYawPitch(landmarks: FloatArray): Pair<Float, Float> {
        val noseX = landmarks[1 * 3]
        val noseY = landmarks[1 * 3 + 1]
        val leftEyeX = landmarks[33 * 3]
        val rightEyeX = landmarks[263 * 3]
        val foreheadY = landmarks[10 * 3 + 1]
        val chinY = landmarks[152 * 3 + 1]

        val distLeft = abs(noseX - leftEyeX)
        val distRight = abs(noseX - rightEyeX)
        val asymmetry = (distRight - distLeft) / (distLeft + distRight + 1e-5f)
        val yawDeg = asymmetry * 60f

        val faceHeight = abs(chinY - foreheadY) + 1e-5f
        val midY = (chinY + foreheadY) / 2.0f
        val pitchOffset = (noseY - midY) / faceHeight
        val pitchDeg = pitchOffset * 70f

        return Pair(yawDeg, pitchDeg)
    }

    fun shutdown() {
        landmarkExecutor.shutdown()
    }
}
