package com.example.pulsefusionbp

import android.media.Image

/**
 * Direct 64x64 Center-Crop Contact PPG Extractor for Rear Camera + LED Torch.
 *
 * Extracts mean [R, G, B] directly from YUV_420_888 planes with zero heap allocation
 * and performs strict optical finger-contact verification so background scenes
 * (desk, air, room light) are never accepted as finger PPG data.
 */
class RearContactProcessor {

    data class ContactPpgFrameResult(
        var meanR: Float = 0f,
        var meanG: Float = 0f,
        var meanB: Float = 0f,
        var isFingerDetected: Boolean = false,
        var isSaturated: Boolean = false,
        var isUnderexposed: Boolean = false
    )

    private val reusable = ContactPpgFrameResult()

    fun processFrame(image: Image): ContactPpgFrameResult {
        val width = image.width
        val height = image.height
        val cropLeft = (width - 64) / 2
        val cropRight = cropLeft + 64
        val cropTop = (height - 64) / 2
        val cropBottom = cropTop + 64
        val pixelCount = 64 * 64

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var satCount = 0

        try {
            for (y in cropTop until cropBottom) {
                val yOffset = y * yRowStride
                val uvOffset = (y shr 1) * uvRowStride
                for (x in cropLeft until cropRight) {
                    val yIdx = yOffset + x
                    val uvIdx = uvOffset + (x shr 1) * uvPixelStride
                    if (yIdx >= yBuffer.limit() || uvIdx >= uBuffer.limit() || uvIdx >= vBuffer.limit()) continue

                    val yVal = yBuffer.get(yIdx).toInt() and 0xFF
                    val uVal = (uBuffer.get(uvIdx).toInt() and 0xFF) - 128
                    val vVal = (vBuffer.get(uvIdx).toInt() and 0xFF) - 128

                    val r = (yVal + 1.402f * vVal).coerceIn(0f, 255f)
                    val g = (yVal - 0.344136f * uVal - 0.714136f * vVal).coerceIn(0f, 255f)
                    val b = (yVal + 1.772f * uVal).coerceIn(0f, 255f)

                    if (r >= 252f && g >= 252f && b >= 252f) {
                        satCount++
                    }
                    sumR += r
                    sumG += g
                    sumB += b
                }
            }
        } catch (e: Throwable) {
            // Buffer may be detached or invalidated during camera transition
            reusable.isFingerDetected = false
            return reusable
        }

        val meanR = (sumR / pixelCount).toFloat()
        val meanG = (sumG / pixelCount).toFloat()
        val meanB = (sumB / pixelCount).toFloat()

        // Strict optical finger contact gate under LED torch transillumination:
        // Human finger tissue over the rear camera + torch produces strong red/amber
        // transmission (meanR >= 75, red dominates blue by >= 1.25x, blue < 185).
        val fingerPresent = (meanR >= 75f) && (meanR > 1.25f * meanB) && (meanB < 185f)

        reusable.meanR = meanR
        reusable.meanG = meanG
        reusable.meanB = meanB
        reusable.isFingerDetected = fingerPresent
        reusable.isSaturated = (satCount.toFloat() / pixelCount) > 0.25f
        reusable.isUnderexposed = meanR < 75f
        return reusable
    }
}
