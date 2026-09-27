package com.dualsite.bp.optical

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Android Camera2 Concurrent Dual-Stream Capture Engine for Poco M6 Pro 5G.
 *
 * Implements concurrent 480p @ 30 FPS streaming across front and rear cameras:
 * - Rear Contact PPG: Manual exposure anti-saturation (ISO 50, 1.0 ms shutter, infinity focus, torch).
 * - Front Facial rPPG: 480p @ 30 FPS with fast color correction.
 * - Hardware monotonic timestamp binding via SENSOR_TIMESTAMP.
 */
class DualCameraEngine(
    private val context: Context,
    private val frameListener: DualStreamFrameListener
) {
    interface DualStreamFrameListener {
        fun onRearContactFrame(image: Image, hardwareTimestampNs: Long)
        fun onFrontFaceFrame(image: Image, hardwareTimestampNs: Long)
    }

    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    // Independent HandlerThreads for parallel zero-contention sensor execution
    private var rearThread: HandlerThread? = null
    private var rearHandler: Handler? = null
    private var frontThread: HandlerThread? = null
    private var frontHandler: Handler? = null

    private var rearCameraDevice: CameraDevice? = null
    private var frontCameraDevice: CameraDevice? = null
    private var rearCaptureSession: CameraCaptureSession? = null
    private var frontCaptureSession: CameraCaptureSession? = null

    private var rearImageReader: ImageReader? = null
    private var frontImageReader: ImageReader? = null

    private val isRecording = AtomicBoolean(false)

    /**
     * Check if hardware HAL supports concurrent streaming of rear (ID "0") and front (ID "1").
     */
    fun checkConcurrentSupport(): Boolean {
        return try {
            val concurrentSets = cameraManager.concurrentCameraIds
            concurrentSets.any { set -> set.contains("0") && set.contains("1") }
        } catch (e: Exception) {
            false
        }
    }

    @SuppressLint("MissingPermission")
    fun startDualStreaming() {
        if (isRecording.getAndSet(true)) return

        rearThread = HandlerThread("RearCameraThread").also { it.start() }
        rearHandler = Handler(rearThread!!.looper)
        frontThread = HandlerThread("FrontCameraThread").also { it.start() }
        frontHandler = Handler(frontThread!!.looper)

        val targetWidth = 640
        val targetHeight = 480

        // 1. Configure Rear ImageReader (maxImages=2 prevents buffer lag and GC pressure)
        rearImageReader = ImageReader.newInstance(targetWidth, targetHeight, ImageFormat.YUV_420_888, 2).apply {
            setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                val timestampNs = image.timestamp
                try {
                    frameListener.onRearContactFrame(image, timestampNs)
                } finally {
                    image.close() // Instant buffer release back to HAL queue
                }
            }, rearHandler)
        }

        // 2. Configure Front ImageReader (maxImages=2)
        frontImageReader = ImageReader.newInstance(targetWidth, targetHeight, ImageFormat.YUV_420_888, 2).apply {
            setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                val timestampNs = image.timestamp
                try {
                    frameListener.onFrontFaceFrame(image, timestampNs)
                } finally {
                    image.close() // Instant buffer release back to HAL queue
                }
            }, frontHandler)
        }

        openRearCamera()
        openFrontCamera()
    }

    @SuppressLint("MissingPermission")
    private fun openRearCamera() {
        cameraManager.openCamera("0", object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                rearCameraDevice = camera
                val surface = rearImageReader!!.surface
                val requestBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(surface)
                    // High-intensity white LED flash active for transillumination
                    set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_TORCH)

                    // Manual Anti-Saturation Exposure Pipeline
                    set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                    set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                    set(CaptureRequest.SENSOR_SENSITIVITY, 50)             // Minimum ISO floor
                    set(CaptureRequest.SENSOR_EXPOSURE_TIME, 1_000_000L)   // 1.0 ms shutter duration

                    // Fixed Infinity Focus (disables Voice Coil Motor hunting)
                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                    set(CaptureRequest.LENS_FOCUS_DISTANCE, 0.0f)

                    // Lock Auto-White Balance
                    set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                    set(CaptureRequest.CONTROL_AWB_LOCK, true)

                    // Disable ISP post-processing filters to preserve raw photometric variance
                    set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
                    set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
                    set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_FAST)
                    set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(30, 30))
                }

                camera.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        rearCaptureSession = session
                        session.setRepeatingRequest(requestBuilder.build(), null, rearHandler)
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {}
                }, rearHandler)
            }
            override fun onDisconnected(camera: CameraDevice) { camera.close() }
            override fun onError(camera: CameraDevice, error: Int) { camera.close() }
        }, rearHandler)
    }

    @SuppressLint("MissingPermission")
    private fun openFrontCamera() {
        cameraManager.openCamera("1", object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                frontCameraDevice = camera
                val surface = frontImageReader!!.surface
                val requestBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(surface)
                    set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                    set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                    set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                    set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(30, 30))
                    set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
                    set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
                    set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_FAST)
                }

                camera.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        frontCaptureSession = session
                        session.setRepeatingRequest(requestBuilder.build(), null, frontHandler)
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {}
                }, frontHandler)
            }
            override fun onDisconnected(camera: CameraDevice) { camera.close() }
            override fun onError(camera: CameraDevice, error: Int) { camera.close() }
        }, frontHandler)
    }

    fun stopStreaming() {
        if (!isRecording.getAndSet(false)) return
        try {
            rearCaptureSession?.stopRepeating()
            rearCaptureSession?.close()
        } catch (e: Exception) {}
        try {
            frontCaptureSession?.stopRepeating()
            frontCaptureSession?.close()
        } catch (e: Exception) {}

        rearCameraDevice?.close()
        frontCameraDevice?.close()
        rearImageReader?.close()
        frontImageReader?.close()

        rearThread?.quitSafely()
        frontThread?.quitSafely()
    }
}
