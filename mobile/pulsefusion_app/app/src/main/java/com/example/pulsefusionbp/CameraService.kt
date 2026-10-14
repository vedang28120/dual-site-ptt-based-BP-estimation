package com.example.pulsefusionbp

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface

class CameraService(
    private val context: Context,
    private val previewSurface: Surface,
    private val frameListener: (Image, Long) -> Unit
) {
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    private var previewRequestBuilder: CaptureRequest.Builder? = null
    private var isTorchEnabled: Boolean = false

    val targetResolution = Size(640, 480)

    fun startBackgroundThread() {
        if (backgroundThread != null) return
        backgroundThread = HandlerThread("Camera2HardwareThread").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e("CameraService", "Error stopping camera thread", e)
        }
    }

    @SuppressLint("MissingPermission")
    fun openCamera(lensFacing: Int = CameraCharacteristics.LENS_FACING_FRONT, enableTorch: Boolean = false) {
        this.isTorchEnabled = enableTorch && (lensFacing == CameraCharacteristics.LENS_FACING_BACK)
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            var selectedCameraId: String? = null
            for (cameraId in manager.cameraIdList) {
                val chars = manager.getCameraCharacteristics(cameraId)
                val facing = chars.get(CameraCharacteristics.LENS_FACING)
                if (facing == lensFacing) {
                    selectedCameraId = cameraId
                    break
                }
            }
            if (selectedCameraId == null && manager.cameraIdList.isNotEmpty()) {
                selectedCameraId = manager.cameraIdList[0]
            }
            selectedCameraId?.let { id ->
                setupImageReader()
                manager.openCamera(id, stateCallback, backgroundHandler)
            }
        } catch (e: CameraAccessException) {
            Log.e("CameraService", "Cannot access camera", e)
        }
    }

    private fun setupImageReader() {
        try {
            imageReader?.setOnImageAvailableListener(null, null)
            imageReader?.close()
        } catch (e: Exception) {}
        imageReader = ImageReader.newInstance(
            targetResolution.width,
            targetResolution.height,
            ImageFormat.YUV_420_888,
            3
        ).apply {
            setOnImageAvailableListener({ reader ->
                val image = try {
                    reader.acquireLatestImage()
                } catch (e: Throwable) {
                    null
                } ?: return@setOnImageAvailableListener

                val hwTimestampNs = image.timestamp
                try {
                    frameListener(image, hwTimestampNs)
                } catch (e: Throwable) {
                    Log.e("CameraService", "Frame listener processing error", e)
                } finally {
                    try {
                        image.close()
                    } catch (e: Throwable) {}
                }
            }, backgroundHandler)
        }
    }

    private val stateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            createCameraPreviewSession()
        }

        override fun onDisconnected(camera: CameraDevice) {
            camera.close()
            cameraDevice = null
        }

        override fun onError(camera: CameraDevice, error: Int) {
            camera.close()
            cameraDevice = null
            Log.e("CameraService", "CameraDevice error: $error")
        }
    }

    private fun createCameraPreviewSession() {
        try {
            val device = cameraDevice ?: return
            val readerSurface = imageReader?.surface ?: return
            previewRequestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(previewSurface)
                addTarget(readerSurface)

                set(CaptureRequest.CONTROL_SCENE_MODE, CaptureRequest.CONTROL_SCENE_MODE_DISABLED)
                set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)
                set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
                set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_FAST)
                set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(30, 30))

                if (isTorchEnabled) {
                    set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_TORCH)
                } else {
                    set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
                }
            }

            device.createCaptureSession(
                listOf(previewSurface, readerSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (cameraDevice == null) return
                        captureSession = session
                        try {
                            previewRequestBuilder?.let { builder ->
                                session.setRepeatingRequest(builder.build(), null, backgroundHandler)
                            }
                        } catch (e: CameraAccessException) {
                            Log.e("CameraService", "Failed to start repeating request", e)
                        }
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e("CameraService", "CaptureSession configuration failed")
                    }
                },
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            Log.e("CameraService", "Error creating capture session", e)
        }
    }

    fun lockExposureAndWhiteBalance() {
        try {
            previewRequestBuilder?.apply {
                set(CaptureRequest.CONTROL_AE_LOCK, true)
                set(CaptureRequest.CONTROL_AWB_LOCK, true)
                captureSession?.setRepeatingRequest(build(), null, backgroundHandler)
            }
        } catch (e: Exception) {
            Log.e("CameraService", "Failed to lock AE/AWB", e)
        }
    }

    fun unlockExposureAndWhiteBalance() {
        try {
            previewRequestBuilder?.apply {
                set(CaptureRequest.CONTROL_AE_LOCK, false)
                set(CaptureRequest.CONTROL_AWB_LOCK, false)
                captureSession?.setRepeatingRequest(build(), null, backgroundHandler)
            }
        } catch (e: Exception) {
            Log.e("CameraService", "Failed to unlock AE/AWB", e)
        }
    }

    fun closeCamera() {
        try {
            imageReader?.setOnImageAvailableListener(null, null)
        } catch (e: Exception) {}
        try {
            captureSession?.stopRepeating()
            captureSession?.close()
        } catch (e: Exception) {}
        captureSession = null
        try {
            cameraDevice?.close()
        } catch (e: Exception) {}
        cameraDevice = null
        try {
            imageReader?.close()
        } catch (e: Exception) {}
        imageReader = null
    }
}
