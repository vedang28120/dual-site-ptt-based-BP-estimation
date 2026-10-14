package com.example.pulsefusionbp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.media.Image
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

data class DualFrameResult(
    val faceRgb15: FloatArray,
    val palmRgb3: FloatArray,
    val isFaceDetected: Boolean,
    val isPalmDetected: Boolean,
    val ear: Float
)

class FaceLandmarkTracker(
    context: Context,
    private val overlayView: OverlayView
) {
    private var faceLandmarker: FaceLandmarker? = null
    private var handLandmarker: HandLandmarker? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()
    private val isDetectingFace = AtomicBoolean(false)
    private val isDetectingHand = AtomicBoolean(false)
    private var lastDetectTime = 0L

    @Volatile private var foreheadHorizontalNorm: RectF? = null
    @Volatile private var leftCheekboneNorm: RectF? = null
    @Volatile private var rightCheekboneNorm: RectF? = null
    @Volatile private var facePointsNorm: FloatArray? = null

    @Volatile var isFaceDetected: Boolean = false
        private set

    @Volatile var isPalmDetected: Boolean = false
        private set

    @Volatile var currentEAR: Float = 0.0f
        private set

    @Volatile var isBlinking: Boolean = false
        private set

    // Normalized Palm ROI mapped from square [0..1] to 480x640 portrait space [0..1]
    // overlayView.palmSquareRect is [0.68..0.96, 0.36..0.64]
    val normPalmPortrait: RectF
        get() = RectF(
            overlayView.palmSquareRect.left,
            0.125f + 0.75f * overlayView.palmSquareRect.top,
            overlayView.palmSquareRect.right,
            0.125f + 0.75f * overlayView.palmSquareRect.bottom
        )

    init {
        setupFaceLandmarker(context)
        setupHandLandmarker(context)
    }

    private fun setupFaceLandmarker(context: Context) {
        try {
            val baseOptionsGpu = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .setDelegate(com.google.mediapipe.tasks.core.Delegate.GPU)
                .build()

            val optionsGpu = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptionsGpu)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.5f)
                .setMinFacePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setResultListener(this::onFaceResults)
                .setErrorListener { e ->
                    Log.e("FaceLandmarkTracker", "FaceLandmarker GPU error: ${e.message}")
                    isDetectingFace.set(false)
                }
                .build()

            faceLandmarker = FaceLandmarker.createFromOptions(context, optionsGpu)
        } catch (gpuEx: Exception) {
            try {
                val baseOptionsCpu = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .setDelegate(com.google.mediapipe.tasks.core.Delegate.CPU)
                .build()

                val optionsCpu = FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(baseOptionsCpu)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setNumFaces(1)
                    .setMinFaceDetectionConfidence(0.5f)
                    .setMinFacePresenceConfidence(0.5f)
                    .setMinTrackingConfidence(0.5f)
                    .setResultListener(this::onFaceResults)
                    .setErrorListener { e ->
                        Log.e("FaceLandmarkTracker", "FaceLandmarker CPU error: ${e.message}")
                        isDetectingFace.set(false)
                    }
                    .build()

                faceLandmarker = FaceLandmarker.createFromOptions(context, optionsCpu)
            } catch (cpuEx: Exception) {
                Log.e("FaceLandmarkTracker", "Failed to init FaceLandmarker", cpuEx)
            }
        }
    }

    private fun setupHandLandmarker(context: Context) {
        try {
            val baseOptionsGpu = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .setDelegate(com.google.mediapipe.tasks.core.Delegate.GPU)
                .build()

            val optionsGpu = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptionsGpu)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.35f)
                .setMinHandPresenceConfidence(0.35f)
                .setMinTrackingConfidence(0.35f)
                .setResultListener(this::onHandResults)
                .setErrorListener { e ->
                    Log.e("FaceLandmarkTracker", "HandLandmarker GPU error: ${e.message}")
                    isDetectingHand.set(false)
                }
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, optionsGpu)
        } catch (gpuEx: Exception) {
            try {
                val baseOptionsCpu = BaseOptions.builder()
                    .setModelAssetPath("hand_landmarker.task")
                    .setDelegate(com.google.mediapipe.tasks.core.Delegate.CPU)
                    .build()

                val optionsCpu = HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptionsCpu)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setNumHands(1)
                    .setMinHandDetectionConfidence(0.35f)
                    .setMinHandPresenceConfidence(0.35f)
                    .setMinTrackingConfidence(0.35f)
                    .setResultListener(this::onHandResults)
                    .setErrorListener { e ->
                        Log.e("FaceLandmarkTracker", "HandLandmarker CPU error: ${e.message}")
                        isDetectingHand.set(false)
                    }
                    .build()

                handLandmarker = HandLandmarker.createFromOptions(context, optionsCpu)
            } catch (cpuEx: Exception) {
                Log.e("FaceLandmarkTracker", "Failed to init HandLandmarker", cpuEx)
            }
        }
    }

    /**
     * Synchronous 30 FPS In-Frame Extraction:
     * Extracts mean [R, G, B] from Horizontal Forehead & Extended Cheekbones (Site 1)
     * and In-Frame Square Palm ROI (Site 2) simultaneously from the same hardware camera frame.
     */
    fun processFrame(
        image: Image,
        isFrontCamera: Boolean = true,
        checkPalm: Boolean = true
    ): DualFrameResult {
        val multiRoiRgb = FloatArray(15)
        val palmRgb = FloatArray(3)

        try {
            val currentForehead = foreheadHorizontalNorm
            val currentLeftCheek = leftCheekboneNorm
            val currentRightCheek = rightCheekboneNorm

            // 1. Extract Facial ROIs: Horizontal Forehead (ROI 0) + Cheekbones (ROI 1, 2)
            if (isFaceDetected && currentForehead != null && currentLeftCheek != null && currentRightCheek != null) {
                val fhRgb = extractMeanRGBFromYUV(image, currentForehead, isFrontCamera)
                val lcRgb = extractMeanRGBFromYUV(image, currentLeftCheek, isFrontCamera)
                val rcRgb = extractMeanRGBFromYUV(image, currentRightCheek, isFrontCamera)

                // Populate 5 facial slots (Forehead, Left Cheek, Right Cheek, Combined Cheeks, Center Forehead)
                multiRoiRgb[0] = fhRgb[0]; multiRoiRgb[1] = fhRgb[1]; multiRoiRgb[2] = fhRgb[2]
                multiRoiRgb[3] = lcRgb[0]; multiRoiRgb[4] = lcRgb[1]; multiRoiRgb[5] = lcRgb[2]
                multiRoiRgb[6] = rcRgb[0]; multiRoiRgb[7] = rcRgb[1]; multiRoiRgb[8] = rcRgb[2]
                multiRoiRgb[9] = (lcRgb[0] + rcRgb[0]) / 2f
                multiRoiRgb[10] = (lcRgb[1] + rcRgb[1]) / 2f
                multiRoiRgb[11] = (lcRgb[2] + rcRgb[2]) / 2f
                multiRoiRgb[12] = fhRgb[0]; multiRoiRgb[13] = fhRgb[1]; multiRoiRgb[14] = fhRgb[2]
            }

            // 2. Extract In-Frame Square Palm ROI
            if (checkPalm) {
                val rawPalmRgb = extractMeanRGBFromYUV(image, normPalmPortrait, isFrontCamera)
                palmRgb[0] = rawPalmRgb[0]
                palmRgb[1] = rawPalmRgb[1]
                palmRgb[2] = rawPalmRgb[2]
            }

            // 3. Update Visual Overlay
            overlayView.updateFaceAndPalm(
                currentForehead,
                currentLeftCheek,
                currentRightCheek,
                facePointsNorm,
                isPalmDetected
            )

            // 4. Asynchronous Neural Detection (~15 FPS without locking 30 FPS camera loop)
            val now = SystemClock.uptimeMillis()
            if (now - lastDetectTime >= 66L) {
                val needFace = isDetectingFace.compareAndSet(false, true)
                val needHand = checkPalm && isDetectingHand.compareAndSet(false, true)

                if (needFace || needHand) {
                    lastDetectTime = now
                    val width = image.width
                    val height = image.height
                    val nv21Bytes = copyYuvToNv21Safe(image)
                    if (nv21Bytes != null) {
                        backgroundExecutor.execute {
                            try {
                                val bitmap = nv21ToBitmap(nv21Bytes, width, height)
                                val matrix = Matrix()
                                if (isFrontCamera) {
                                    matrix.postRotate(270f)
                                    matrix.postScale(-1f, 1f)
                                } else {
                                    matrix.postRotate(90f)
                                }
                                val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true)
                                bitmap.recycle()

                                val mpImage = BitmapImageBuilder(rotatedBitmap).build()
                                if (needFace) {
                                    faceLandmarker?.detectAsync(mpImage, now)
                                }
                                if (needHand) {
                                    handLandmarker?.detectAsync(mpImage, now)
                                }
                            } catch (e: Throwable) {
                                Log.w("FaceLandmarkTracker", "Asynchronous detection error: ${e.message}")
                                if (needFace) isDetectingFace.set(false)
                                if (needHand) isDetectingHand.set(false)
                            }
                        }
                    } else {
                        if (needFace) isDetectingFace.set(false)
                        if (needHand) isDetectingHand.set(false)
                    }
                }
            }
        } catch (e: Throwable) {
            Log.w("FaceLandmarkTracker", "processFrame safe catch: ${e.message}")
        }

        return DualFrameResult(
            faceRgb15 = multiRoiRgb,
            palmRgb3 = palmRgb,
            isFaceDetected = isFaceDetected,
            isPalmDetected = isPalmDetected,
            ear = currentEAR
        )
    }

    private fun onFaceResults(result: FaceLandmarkerResult, input: com.google.mediapipe.framework.image.MPImage) {
        try {
            if (result.faceLandmarks().isNotEmpty()) {
                val landmarks = result.faceLandmarks()[0]

                fun calcEAR(indices: IntArray): Float {
                    val p1 = landmarks[indices[0]]
                    val p2 = landmarks[indices[1]]
                    val p3 = landmarks[indices[2]]
                    val p4 = landmarks[indices[3]]
                    val p5 = landmarks[indices[4]]
                    val p6 = landmarks[indices[5]]
                    val v1 = hypot((p2.x() - p6.x()).toDouble(), (p2.y() - p6.y()).toDouble())
                    val v2 = hypot((p3.x() - p5.x()).toDouble(), (p3.y() - p5.y()).toDouble())
                    val h = hypot((p1.x() - p4.x()).toDouble(), (p1.y() - p4.y()).toDouble())
                    return if (h > 1e-5) ((v1 + v2) / (2.0 * h)).toFloat() else 0.0f
                }

                val leftEAR = calcEAR(intArrayOf(33, 160, 158, 133, 153, 144))
                val rightEAR = calcEAR(intArrayOf(362, 385, 387, 263, 373, 380))
                currentEAR = (leftEAR + rightEAR) / 2.0f
                isBlinking = currentEAR < 0.20f

                // EXACT FACIAL LANDMARK ROI COORDINATE POINTS SPECIFIED BY USER:
                // 1. Forehead Zone: 10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109
                val foreheadIndices = intArrayOf(10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109)

                // 2. Left Cheek Zone: 346, 347, 348, 329, 371, 425, 426, 352, 345
                val leftCheekIndices = intArrayOf(346, 347, 348, 329, 371, 425, 426, 352, 345)

                // 3. Right Cheek Zone: 117, 118, 119, 100, 142, 205, 206, 123, 116
                val rightCheekIndices = intArrayOf(117, 118, 119, 100, 142, 205, 206, 123, 116)

                val targetFh = getBoundingBox(landmarks, foreheadIndices)
                val targetLc = getBoundingBox(landmarks, leftCheekIndices)
                val targetRc = getBoundingBox(landmarks, rightCheekIndices)

                foreheadHorizontalNorm = glideBox(foreheadHorizontalNorm, targetFh)
                leftCheekboneNorm = glideBox(leftCheekboneNorm, targetLc)
                rightCheekboneNorm = glideBox(rightCheekboneNorm, targetRc)

                val pts = FloatArray(landmarks.size * 2)
                for (i in landmarks.indices) {
                    pts[i * 2] = landmarks[i].x()
                    pts[i * 2 + 1] = landmarks[i].y()
                }
                facePointsNorm = pts
                isFaceDetected = true
            } else {
                isFaceDetected = false
                foreheadHorizontalNorm = null
                leftCheekboneNorm = null
                rightCheekboneNorm = null
                facePointsNorm = null
            }
        } finally {
            isDetectingFace.set(false)
        }
    }

    private fun onHandResults(result: HandLandmarkerResult, input: com.google.mediapipe.framework.image.MPImage) {
        try {
            if (result.landmarks().isNotEmpty()) {
                val hand = result.landmarks()[0]
                var handMeanX = 0f
                var handMeanY = 0f
                for (lm in hand) {
                    handMeanX += lm.x()
                    handMeanY += lm.y()
                }
                handMeanX /= hand.size
                handMeanY /= hand.size

                // Hand must be on the side of the preview (x >= 0.45, y in 0.20..0.80) to count as Site 2
                isPalmDetected = (handMeanX >= 0.45f && handMeanY in 0.20f..0.80f)
            } else {
                isPalmDetected = false
            }
        } finally {
            isDetectingHand.set(false)
        }
    }

    private fun getBoundingBox(
        landmarks: List<NormalizedLandmark>,
        indices: IntArray
    ): RectF {
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var maxY = Float.MIN_VALUE
        for (idx in indices) {
            val lm = landmarks[idx]
            val x = lm.x().coerceIn(0.01f, 0.99f)
            val y = lm.y().coerceIn(0.01f, 0.99f)
            if (x < minX) minX = x
            if (y < minY) minY = y
            if (x > maxX) maxX = x
            if (y > maxY) maxY = y
        }
        return RectF(minX, minY, maxX, maxY)
    }

    private fun glideBox(current: RectF?, target: RectF, alpha: Float = 0.35f): RectF {
        if (current == null) return RectF(target)
        return RectF(
            current.left + (target.left - current.left) * alpha,
            current.top + (target.top - current.top) * alpha,
            current.right + (target.right - current.right) * alpha,
            current.bottom + (target.bottom - current.bottom) * alpha
        )
    }

    private fun extractMeanRGBFromYUV(image: Image, normRect: RectF, isFrontCamera: Boolean): FloatArray {
        try {
            val width = image.width
            val height = image.height

            val yPlane = image.planes[0]
            val uPlane = image.planes[1]
            val vPlane = image.planes[2]

            val yBuffer = yPlane.buffer
            val uBuffer = uPlane.buffer
            val vBuffer = vPlane.buffer

            val yRowStride = yPlane.rowStride
            val uvRowStride = uPlane.rowStride
            val uvPixelStride = uPlane.pixelStride

            val rawXMin = if (isFrontCamera) (1.0f - normRect.bottom) else normRect.top
            val rawXMax = if (isFrontCamera) (1.0f - normRect.top) else normRect.bottom
            val rawYMin = 1.0f - normRect.right
            val rawYMax = 1.0f - normRect.left

            val startX = (min(rawXMin, rawXMax) * width).toInt().coerceIn(0, width - 1)
            val endX = (max(rawXMin, rawXMax) * width).toInt().coerceIn(0, width - 1)
            val startY = (min(rawYMin, rawYMax) * height).toInt().coerceIn(0, height - 1)
            val endY = (max(rawYMin, rawYMax) * height).toInt().coerceIn(0, height - 1)

            var rSum = 0L
            var gSum = 0L
            var bSum = 0L
            var count = 0

            for (y in startY..endY step 2) {
                val yOffset = y * yRowStride
                val uvOffset = (y shr 1) * uvRowStride
                for (x in startX..endX step 2) {
                    val yIdx = yOffset + x
                    val uvIdx = uvOffset + (x shr 1) * uvPixelStride
                    if (yIdx >= yBuffer.limit() || uvIdx >= uBuffer.limit() || uvIdx >= vBuffer.limit()) continue

                    val yVal = (yBuffer.get(yIdx).toInt() and 0xFF)
                    val uVal = (uBuffer.get(uvIdx).toInt() and 0xFF) - 128
                    val vVal = (vBuffer.get(uvIdx).toInt() and 0xFF) - 128

                    val r = (yVal + 1.370705f * vVal).toInt().coerceIn(0, 255)
                    val g = (yVal - 0.337633f * uVal - 0.698001f * vVal).toInt().coerceIn(0, 255)
                    val b = (yVal + 1.732446f * uVal).toInt().coerceIn(0, 255)

                    rSum += r
                    gSum += g
                    bSum += b
                    count++
                }
            }

            if (count == 0) return floatArrayOf(0f, 0f, 0f)
            return floatArrayOf(rSum.toFloat() / count, gSum.toFloat() / count, bSum.toFloat() / count)
        } catch (e: Throwable) {
            return floatArrayOf(0f, 0f, 0f)
        }
    }

    private fun copyYuvToNv21Safe(image: Image): ByteArray? {
        return try {
            val width = image.width
            val height = image.height
            val nv21 = ByteArray(width * height * 3 / 2)

            val yPlane = image.planes[0]
            val yBuffer = yPlane.buffer
            val yRowStride = yPlane.rowStride
            var pos = 0

            if (yRowStride == width) {
                yBuffer.get(nv21, 0, width * height)
                pos = width * height
            } else {
                val yBytes = ByteArray(yBuffer.remaining())
                yBuffer.get(yBytes)
                for (r in 0 until height) {
                    System.arraycopy(yBytes, r * yRowStride, nv21, pos, width)
                    pos += width
                }
            }

            val uPlane = image.planes[1]
            val vPlane = image.planes[2]
            val vBuffer = vPlane.buffer
            val uBuffer = uPlane.buffer
            val uvRowStride = vPlane.rowStride
            val uvPixelStride = uPlane.pixelStride

            val vBytes = ByteArray(vBuffer.remaining())
            vBuffer.get(vBytes)
            val uBytes = ByteArray(uBuffer.remaining())
            uBuffer.get(uBytes)

            for (r in 0 until height / 2) {
                val rowOffset = r * uvRowStride
                for (c in 0 until width / 2) {
                    val uvIdx = rowOffset + c * uvPixelStride
                    if (uvIdx < vBytes.size && uvIdx < uBytes.size) {
                        nv21[pos++] = vBytes[uvIdx]
                        nv21[pos++] = uBytes[uvIdx]
                    }
                }
            }

            yBuffer.rewind()
            uBuffer.rewind()
            vBuffer.rewind()

            nv21
        } catch (e: Throwable) {
            Log.w("FaceLandmarkTracker", "copyYuvToNv21Safe error: ${e.message}")
            null
        }
    }

    private fun nv21ToBitmap(nv21: ByteArray, width: Int, height: Int): Bitmap {
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 85, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    fun close() {
        try {
            backgroundExecutor.shutdown()
            faceLandmarker?.close()
            handLandmarker?.close()
        } catch (e: Throwable) {}
    }
}
