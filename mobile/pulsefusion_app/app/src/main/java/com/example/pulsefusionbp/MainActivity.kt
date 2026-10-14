package com.example.pulsefusionbp

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.media.Image
import android.os.Bundle
import android.os.Environment
import android.text.InputType
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.example.pulsefusionbp.databinding.ActivityMainBinding
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity(), TextureView.SurfaceTextureListener, SensorEventListener {

    enum class AppState {
        IDLE,
        CALIBRATING, // 0.1s AE/AWB settling window for all modes
        RECORDING,
        PROCESSING,
        RESULT
    }

    enum class ScanMode {
        DUAL_SITE_SIMULTANEOUS,
        FACE_ONLY,
        FINGER_ONLY
    }

    private lateinit var binding: ActivityMainBinding
    private var cameraService: CameraService? = null
    private var faceTracker: FaceLandmarkTracker? = null
    private val rearContactProcessor = RearContactProcessor()
    private lateinit var prefs: SharedPreferences

    private var currentLensFacing = CameraCharacteristics.LENS_FACING_FRONT
    private var appState = AppState.IDLE
    private var scanMode = ScanMode.DUAL_SITE_SIMULTANEOUS
    private var activeModelTier = "pf_50m" // "pf_50m" (50.94M ONNX), "xl" (1.48M TCN), or "compact" (231k TCN)
    private lateinit var pulseFormerEngine: PulseFormerOnnxEngine

    // User Profile Demographics
    private var userHeightCm: Int = 175
    private var userAge: Int = 28

    // Hardware frame buffers
    private val faceRgbBuffer = mutableListOf<Float>()
    private val faceTsBuffer = mutableListOf<Float>()
    private val fingerRgbBuffer = mutableListOf<Float>() // Holds Palm RGB in Dual-Site, Finger in Finger-Only
    private val fingerTsBuffer = mutableListOf<Float>()
    private val imuGravityZBuffer = mutableListOf<Float>()

    private val targetFrames = 210 // ~7.0 seconds at 30 FPS
    private var calibrationStartTime = 0L
    private var isExposureLocked = false

    // Frame rate & interval
    private var lastHwTimestampNs = 0L
    private var liveFps = 0.0f
    private var liveIntervalMs = 0.0f

    // 6-Axis IMU for phone stability metric
    private var sensorManager: SensorManager? = null
    private var gravitySensor: Sensor? = null
    private var latestGravityX = 0f
    private var latestGravityY = 0f
    private var latestGravityZ = 0f

    // Live UI throttle
    private var lastUiUpdateMs = 0L
    private var lastProcessedResultJson: JSONObject? = null

    private val coroutineScope = CoroutineScope(Dispatchers.Main + Job())

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            setupCameraIfReady()
        } else {
            Toast.makeText(this, "Camera permission is strictly required for optical pulse scan.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        prefs = getSharedPreferences("PulseFusionPrefs", Context.MODE_PRIVATE)
        userHeightCm = prefs.getInt("height_cm", 175)
        userAge = prefs.getInt("age", 28)

        // Initialize Chaquopy Python Runtime
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        // Initialize Native PulseFormer-Lite-50M ONNX Runtime Mobile Engine
        pulseFormerEngine = PulseFormerOnnxEngine(this)

        faceTracker = FaceLandmarkTracker(this, binding.overlayView)

        setupChart()
        setupSensors()
        setupClickListeners()
        updateProfileDisplay()
        updateModeUI()
        binding.btnModelToggle.text = "PF-50M"

        binding.viewFinder.surfaceTextureListener = this

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            setupCameraIfReady()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupSensors() {
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        gravitySensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GRAVITY)
            ?: sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    }

    override fun onResume() {
        super.onResume()
        gravitySensor?.let { sensorManager?.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
        cameraService?.startBackgroundThread()
    }

    override fun onPause() {
        super.onPause()
        sensorManager?.unregisterListener(this)
        cameraService?.closeCamera()
        cameraService?.stopBackgroundThread()
    }

    override fun onDestroy() {
        super.onDestroy()
        faceTracker?.close()
        pulseFormerEngine.close()
    }

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        configureSquareTransform(width, height)
        setupCameraIfReady()
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
        configureSquareTransform(width, height)
    }

    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}

    /**
     * Center crops the 480x640 portrait camera feed into a 480x480 square inside the 1:1 viewFinder,
     * maintaining 1:1 square pixel geometry with zero horizontal or vertical distortion.
     */
    private fun configureSquareTransform(width: Int, height: Int) {
        if (width <= 0 || height <= 0) return
        val matrix = Matrix()
        val scaleY = 640.0f / 480.0f
        matrix.postScale(1.0f, scaleY, width / 2.0f, height / 2.0f)
        binding.viewFinder.setTransform(matrix)
    }

    private fun setupCameraIfReady() {
        if (!binding.viewFinder.isAvailable) return
        val surfaceTexture = binding.viewFinder.surfaceTexture ?: return
        surfaceTexture.setDefaultBufferSize(640, 480)
        val previewSurface = Surface(surfaceTexture)

        cameraService?.closeCamera()
        cameraService?.stopBackgroundThread()

        cameraService = CameraService(this, previewSurface) { image, hwTimestampNs ->
            onCameraFrame(image, hwTimestampNs)
        }.also {
            it.startBackgroundThread()
            val enableTorch = (scanMode == ScanMode.FINGER_ONLY && currentLensFacing == CameraCharacteristics.LENS_FACING_BACK)
            it.openCamera(currentLensFacing, enableTorch = enableTorch)
        }
        configureSquareTransform(binding.viewFinder.width, binding.viewFinder.height)
    }

    private fun setupChart() {
        binding.bvpChart.apply {
            description.isEnabled = false
            setTouchEnabled(true)
            isDragEnabled = true
            setScaleEnabled(true)
            setPinchZoom(true)
            setDrawGridBackground(false)
            setBackgroundColor(Color.parseColor("#161616"))
            legend.textColor = Color.parseColor("#E0E0E0")
            legend.textSize = 9.0f

            xAxis.apply {
                position = XAxis.XAxisPosition.BOTTOM
                textColor = Color.parseColor("#888888")
                setDrawGridLines(false)
                textSize = 8.5f
                axisMinimum = 0f
                axisMaximum = targetFrames.toFloat()
            }

            axisLeft.apply {
                textColor = Color.parseColor("#888888")
                setDrawGridLines(true)
                gridColor = Color.parseColor("#262626")
                textSize = 8.5f
            }
            axisRight.isEnabled = false
            setNoDataText("WAITING FOR SCAN")
            setNoDataTextColor(Color.parseColor("#888888"))
        }
    }

    private fun setupClickListeners() {
        binding.btnStartScan.setOnClickListener {
            handleStartScanClick()
        }

        binding.btnHelp.setOnClickListener {
            showHelpGuideDialog()
        }

        binding.btnModeToggle.setOnClickListener {
            if (appState != AppState.IDLE && appState != AppState.RESULT) return@setOnClickListener
            // Cycle through: DUAL_SITE -> FACE_ONLY -> FINGER_ONLY -> DUAL_SITE
            scanMode = when (scanMode) {
                ScanMode.DUAL_SITE_SIMULTANEOUS -> ScanMode.FACE_ONLY
                ScanMode.FACE_ONLY -> ScanMode.FINGER_ONLY
                ScanMode.FINGER_ONLY -> ScanMode.DUAL_SITE_SIMULTANEOUS
            }
            updateModeUI()
            resetSession(keepPreview = true)

            if (scanMode == ScanMode.FINGER_ONLY) {
                currentLensFacing = CameraCharacteristics.LENS_FACING_BACK
                setupCameraIfReady()
            } else if (currentLensFacing == CameraCharacteristics.LENS_FACING_BACK && scanMode != ScanMode.FINGER_ONLY) {
                currentLensFacing = CameraCharacteristics.LENS_FACING_FRONT
                setupCameraIfReady()
            }
        }

        binding.btnModelToggle.setOnClickListener {
            activeModelTier = when (activeModelTier) {
                "pf_50m" -> "xl"
                "xl" -> "compact"
                else -> "pf_50m"
            }
            binding.btnModelToggle.text = when (activeModelTier) {
                "pf_50m" -> "PF-50M"
                "xl" -> "TCN-XL"
                else -> "COMPACT"
            }
            val modelName = when (activeModelTier) {
                "pf_50m" -> "PulseFormer-Lite-50M (50.94M params, ONNX INT8)"
                "xl" -> "DualStreamTCN-XL (1.48M params)"
                else -> "DualStreamTCN-Compact (231k params)"
            }
            Toast.makeText(this, "Active Model: $modelName", Toast.LENGTH_SHORT).show()
        }

        binding.btnSwitchCamera.setOnClickListener {
            if (appState != AppState.IDLE && appState != AppState.RESULT) return@setOnClickListener
            currentLensFacing = if (currentLensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
                CameraCharacteristics.LENS_FACING_BACK
            } else {
                CameraCharacteristics.LENS_FACING_FRONT
            }
            setupCameraIfReady()
        }

        binding.btnProfile.setOnClickListener {
            showProfileEditDialog()
        }

        binding.btnExportCsv.setOnClickListener {
            exportSessionCsv()
        }
    }

    private fun updateModeUI() {
        when (scanMode) {
            ScanMode.DUAL_SITE_SIMULTANEOUS -> {
                binding.btnModeToggle.text = "DUAL-SITE"
                binding.btnStartScan.text = "START DUAL-SITE SCAN (FACE + PALM)"
                binding.tvDiagnosisBadge.text = "IDLE — ALIGN FACE IN CENTER & PALM IN SQUARE"
                binding.overlayView.setOverlayMode(OverlayView.OverlayMode.DUAL_SITE)
            }
            ScanMode.FACE_ONLY -> {
                binding.btnModeToggle.text = "FACE-ONLY"
                binding.btnStartScan.text = "START FACE rPPG SCAN"
                binding.tvDiagnosisBadge.text = "IDLE — ALIGN FACE IN VIEWFINDER"
                binding.overlayView.setOverlayMode(OverlayView.OverlayMode.FACE_ONLY)
            }
            ScanMode.FINGER_ONLY -> {
                binding.btnModeToggle.text = "FINGER-ONLY"
                binding.btnStartScan.text = "START REAR FINGER PPG SCAN"
                binding.tvDiagnosisBadge.text = "IDLE — PLACE FINGER OVER REAR LENS & FLASH"
                binding.overlayView.setOverlayMode(OverlayView.OverlayMode.FINGER_ONLY)
            }
        }
    }

    private fun updateProfileDisplay() {
        binding.btnProfile.text = "${userHeightCm}cm | ${userAge}y"
        val headElevCm = String.format(Locale.US, "%.1f", 0.18f * userHeightCm)
        binding.tvSubHeader.text = "Dual-Site PTT | Head Elev: +$headElevCm cm"
    }

    private fun showProfileEditDialog() {
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(50, 40, 50, 20)
        }

        val heightInput = EditText(this).apply {
            hint = "Height in cm (e.g. 175)"
            inputType = InputType.TYPE_CLASS_NUMBER
            setText(userHeightCm.toString())
        }

        val ageInput = EditText(this).apply {
            hint = "Age in years (e.g. 28)"
            inputType = InputType.TYPE_CLASS_NUMBER
            setText(userAge.toString())
        }

        layout.addView(heightInput)
        layout.addView(ageInput)

        AlertDialog.Builder(this)
            .setTitle("User Demographics")
            .setMessage("Your height is used to calculate your exact arterial path length (d_path = 0.6736 * height) and head elevation.")
            .setView(layout)
            .setPositiveButton("Save") { _, _ ->
                val h = heightInput.text.toString().toIntOrNull() ?: userHeightCm
                val a = ageInput.text.toString().toIntOrNull() ?: userAge
                userHeightCm = h.coerceIn(120, 220)
                userAge = a.coerceIn(16, 95)
                prefs.edit().putInt("height_cm", userHeightCm).putInt("age", userAge).apply()
                updateProfileDisplay()
                Toast.makeText(this, "Profile Saved: ${userHeightCm} cm, ${userAge} y", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showHelpGuideDialog() {
        val scrollView = ScrollView(this)
        val tv = TextView(this).apply {
            setPadding(50, 30, 50, 30)
            setTextColor(Color.parseColor("#DDDDDD"))
            textSize = 13.5f
            setLineSpacing(6f, 1.15f)
            text = """
                PULSEFUSION BP v2.0 — USER GUIDE & INSTRUCTIONS

                1. OVERVIEW:
                PulseFusion BP provides continuous, cuffless blood pressure estimation using dual-site photoplethysmography (PPG) and deep DualStream Temporal Convolutional Networks (DS-TCN) combined with hemodynamics physics.

                2. TEST MODES:
                • MODE 1: DUAL-SITE SIMULTANEOUS (FACE + PALM)
                  - Hold your phone in front of you.
                  - Keep your face centered (Site 1: Carotid rPPG).
                  - Raise your open palm beside your face inside the square guide box (Site 2: Palmar rPPG).
                  - Both 'FACE: OK' and 'PALM: OK' must be detected for the test to begin.
                  - Measures the exact beat-by-beat transit time (ΔPTT) between face and hand from the same 30 FPS video frames!

                • MODE 2: FACE-ONLY rPPG
                  - Align your face in the camera frame.
                  - Measures facial arterial pulse wave contour (k_form, rise time, PWV).

                • MODE 3: FINGER-ONLY CONTACT PPG
                  - Place your index fingertip gently over the rear camera lens and LED flash.
                  - The app pauses 0.1s for auto-exposure to adapt to tissue transillumination, then locks exposure before recording.

                3. WHY HEIGHT & DEMOGRAPHICS MATTER:
                  - Pulse wave velocity (PWV) is calculated as:
                    PWV = Path Length / Transit Time
                  - Path length is anatomically proportional to body height:
                    d_path = 0.6736 × Height
                  - In Dual-Site mode, both face and hand are elevated above the heart by +0.18 × Height.

                4. BEST PRACTICES FOR MAXIMUM ACCURACY:
                  - Stay seated, relaxed, and avoid talking during the 7-second scan.
                  - Keep your hand still inside the square guide box.
                  - Normal indoor room lighting is ideal.
            """.trimIndent()
        }
        scrollView.addView(tv)

        AlertDialog.Builder(this)
            .setTitle("Instructions & Guide")
            .setView(scrollView)
            .setPositiveButton("Close", null)
            .show()
    }

    private fun handleStartScanClick() {
        if (appState == AppState.PROCESSING) return

        if (appState == AppState.RECORDING || appState == AppState.CALIBRATING) {
            resetSession(keepPreview = true)
            binding.tvDiagnosisBadge.text = "SCAN CANCELLED — READY"
            return
        }

        // Fresh Start: Clean chart so no old data is carried forward!
        resetSession(keepPreview = false)

        appState = AppState.CALIBRATING
        calibrationStartTime = System.currentTimeMillis()
        isExposureLocked = false
        binding.progressBar.visibility = View.VISIBLE
        binding.progressBar.progress = 0
        binding.btnStartScan.text = "CANCEL SCAN"

        when (scanMode) {
            ScanMode.DUAL_SITE_SIMULTANEOUS -> {
                binding.tvDiagnosisBadge.text = "ALIGN BOTH FACE IN CENTER & PALM IN SQUARE..."
            }
            ScanMode.FACE_ONLY -> {
                binding.tvDiagnosisBadge.text = "ALIGN FACE IN VIEWFINDER..."
            }
            ScanMode.FINGER_ONLY -> {
                binding.tvDiagnosisBadge.text = "PLACE FINGER OVER REAR LENS & FLASH..."
            }
        }
    }

    private fun resetSession(keepPreview: Boolean) {
        appState = AppState.IDLE
        faceRgbBuffer.clear()
        faceTsBuffer.clear()
        fingerRgbBuffer.clear()
        fingerTsBuffer.clear()
        imuGravityZBuffer.clear()
        isExposureLocked = false

        binding.progressBar.visibility = View.GONE
        binding.progressBar.progress = 0
        binding.loadingSpinner.visibility = View.GONE
        updateModeUI()

        cameraService?.unlockExposureAndWhiteBalance()

        // Clear chart completely: zero carry-forward lines!
        binding.bvpChart.clear()
        binding.bvpChart.data = null
        binding.bvpChart.invalidate()
    }

    /**
     * Primary Hardware Camera Callback running on background Camera2 thread.
     */
    private fun onCameraFrame(image: Image, hwTimestampNs: Long) {
        val nowMs = System.currentTimeMillis()

        if (lastHwTimestampNs > 0L) {
            val deltaNs = hwTimestampNs - lastHwTimestampNs
            if (deltaNs > 0) {
                liveIntervalMs = deltaNs / 1_000_000.0f
                liveFps = 0.9f * liveFps + 0.1f * (1_000_000_000.0f / deltaNs)
            }
        }
        lastHwTimestampNs = hwTimestampNs
        val timestampMs = hwTimestampNs / 1_000_000.0f
        val isFront = (currentLensFacing == CameraCharacteristics.LENS_FACING_FRONT)

        if (scanMode == ScanMode.FINGER_ONLY) {
            val cropResult = rearContactProcessor.processFrame(image)
            val fingerDetected = cropResult.isFingerDetected
            binding.overlayView.updateRearCropOverlay(fingerDetected)

            when (appState) {
                AppState.CALIBRATING -> {
                    if (fingerDetected) {
                        if (nowMs - calibrationStartTime >= 100L) {
                            cameraService?.lockExposureAndWhiteBalance()
                            isExposureLocked = true
                            appState = AppState.RECORDING
                            runOnUiThread {
                                binding.tvDiagnosisBadge.text = "RECORDING FINGER PPG (KEEP STILL)..."
                            }
                        }
                    } else {
                        calibrationStartTime = nowMs
                        runOnUiThread {
                            binding.tvDiagnosisBadge.text = "PLACE FINGER OVER REAR LENS & FLASH"
                        }
                    }
                }
                AppState.RECORDING -> {
                    if (fingerDetected) {
                        fingerRgbBuffer.add(cropResult.meanR)
                        fingerRgbBuffer.add(cropResult.meanG)
                        fingerRgbBuffer.add(cropResult.meanB)
                        fingerTsBuffer.add(timestampMs)
                        val captured = fingerTsBuffer.size

                        runOnUiThread {
                            binding.progressBar.progress = captured
                            binding.tvDiagnosisBadge.text = "RECORDING FINGER PPG ($captured/$targetFrames frames)..."
                            appendSingleChartPoint(captured.toFloat(), cropResult.meanG, isFace = false)
                        }

                        if (captured >= targetFrames) {
                            onScanComplete()
                        }
                    } else {
                        runOnUiThread {
                            binding.tvDiagnosisBadge.text = "FINGER LIFTED — HOLD FINGER GENTLY OVER LENS"
                        }
                    }
                }
                else -> {}
            }

            if (nowMs - lastUiUpdateMs >= 80L) {
                lastUiUpdateMs = nowMs
                runOnUiThread {
                    val aeStatus = if (isExposureLocked) "LOCKED" else "SETTLING"
                    val contactStr = if (fingerDetected) "CONTACT OK" else "NOT COVERED"
                    binding.tvLiveSensors.text =
                        "CAM: REAR ($contactStr) | FPS: ${String.format(Locale.US, "%.1f", liveFps)} | AE: $aeStatus"
                }
            }

        } else {
            // DUAL-SITE SIMULTANEOUS OR FACE-ONLY MODE
            val checkPalm = (scanMode == ScanMode.DUAL_SITE_SIMULTANEOUS)
            val frameResult = faceTracker?.processFrame(image, isFrontCamera = isFront, checkPalm = checkPalm)
                ?: DualFrameResult(FloatArray(15), FloatArray(3), false, false, 0f)

            val faceDetected = frameResult.isFaceDetected
            val palmDetected = frameResult.isPalmDetected
            val ear = frameResult.ear
            val foreheadG = frameResult.faceRgb15[1]
            val palmG = frameResult.palmRgb3[1]

            // Strict Dual-Detection Gate: BOTH Face and Palm must be detected for Dual-Site!
            val targetReady = if (scanMode == ScanMode.DUAL_SITE_SIMULTANEOUS) {
                (faceDetected && palmDetected)
            } else {
                faceDetected
            }

            when (appState) {
                AppState.CALIBRATING -> {
                    if (targetReady) {
                        if (nowMs - calibrationStartTime >= 100L) {
                            cameraService?.lockExposureAndWhiteBalance()
                            isExposureLocked = true
                            appState = AppState.RECORDING
                            runOnUiThread {
                                val msg = if (scanMode == ScanMode.DUAL_SITE_SIMULTANEOUS) {
                                    "RECORDING DUAL-SITE (FACE + PALM) — HOLD STILL..."
                                } else {
                                    "RECORDING FACE rPPG — HOLD STILL..."
                                }
                                binding.tvDiagnosisBadge.text = msg
                            }
                        }
                    } else {
                        calibrationStartTime = nowMs
                        runOnUiThread {
                            val waitMsg = if (scanMode == ScanMode.DUAL_SITE_SIMULTANEOUS) {
                                when {
                                    !faceDetected && !palmDetected -> "ALIGN FACE IN CENTER & PALM IN SQUARE"
                                    faceDetected && !palmDetected -> "FACE: OK — NOW RAISE PALM IN SQUARE (SITE 2)"
                                    !faceDetected && palmDetected -> "PALM: OK — ALIGN FACE IN CENTER (SITE 1)"
                                    else -> "SETTLING EXPOSURE..."
                                }
                            } else {
                                "ALIGN FACE IN CENTER"
                            }
                            binding.tvDiagnosisBadge.text = waitMsg
                        }
                    }
                }
                AppState.RECORDING -> {
                    if (targetReady) {
                        for (f in frameResult.faceRgb15) faceRgbBuffer.add(f)
                        faceTsBuffer.add(timestampMs)

                        if (scanMode == ScanMode.DUAL_SITE_SIMULTANEOUS) {
                            for (p in frameResult.palmRgb3) fingerRgbBuffer.add(p)
                            fingerTsBuffer.add(timestampMs)
                        }

                        val captured = faceTsBuffer.size

                        runOnUiThread {
                            binding.progressBar.progress = captured
                            binding.tvDiagnosisBadge.text = "RECORDING ($captured/$targetFrames frames)..."
                            if (scanMode == ScanMode.DUAL_SITE_SIMULTANEOUS) {
                                appendDualChartPoint(captured.toFloat(), foreheadG, palmG)
                            } else {
                                appendSingleChartPoint(captured.toFloat(), foreheadG, isFace = true)
                            }
                        }

                        if (captured >= targetFrames) {
                            onScanComplete()
                        }
                    } else {
                        runOnUiThread {
                            binding.tvDiagnosisBadge.text = "TARGET LOST — KEEP BOTH FACE & PALM IN VIEW"
                        }
                    }
                }
                else -> {}
            }

            if (nowMs - lastUiUpdateMs >= 80L) {
                lastUiUpdateMs = nowMs
                runOnUiThread {
                    val aeStatus = if (isExposureLocked) "LOCKED" else "SETTLING"
                    val faceStr = if (faceDetected) "OK (EAR ${String.format(Locale.US, "%.2f", ear)})" else "WAITING"
                    val palmStr = if (palmDetected) "OK" else "WAITING"
                    val camName = if (isFront) "FRONT" else "REAR"
                    binding.tvLiveSensors.text =
                        "CAM: $camName | FPS: ${String.format(Locale.US, "%.1f", liveFps)} | AE: $aeStatus | FACE: $faceStr | PALM: $palmStr"
                }
            }
        }
    }

    private fun appendDualChartPoint(xVal: Float, faceGreen: Float, palmGreen: Float) {
        val lineData = binding.bvpChart.data ?: LineData().also { binding.bvpChart.data = it }

        var faceSet = lineData.getDataSetByLabel("Site 1: Face rPPG", true) as? LineDataSet
        if (faceSet == null) {
            faceSet = LineDataSet(mutableListOf(), "Site 1: Face rPPG").apply {
                color = Color.parseColor("#38BDF8")
                setDrawCircles(false)
                lineWidth = 1.8f
                setDrawValues(false)
                mode = LineDataSet.Mode.CUBIC_BEZIER
            }
            lineData.addDataSet(faceSet)
        }

        var palmSet = lineData.getDataSetByLabel("Site 2: Palm rPPG", true) as? LineDataSet
        if (palmSet == null) {
            palmSet = LineDataSet(mutableListOf(), "Site 2: Palm rPPG").apply {
                color = Color.parseColor("#F59E0B")
                setDrawCircles(false)
                lineWidth = 1.8f
                setDrawValues(false)
                mode = LineDataSet.Mode.CUBIC_BEZIER
            }
            lineData.addDataSet(palmSet)
        }

        faceSet.addEntry(Entry(xVal, faceGreen))
        palmSet.addEntry(Entry(xVal, palmGreen))

        lineData.notifyDataChanged()
        binding.bvpChart.notifyDataSetChanged()
        binding.bvpChart.invalidate()
    }

    private fun appendSingleChartPoint(xVal: Float, yVal: Float, isFace: Boolean) {
        val lineData = binding.bvpChart.data ?: LineData().also { binding.bvpChart.data = it }
        val label = if (isFace) "Site 1: Face rPPG" else "Site 2: Finger Contact PPG"
        val color = if (isFace) Color.parseColor("#38BDF8") else Color.parseColor("#F59E0B")

        var dataSet = lineData.getDataSetByLabel(label, true) as? LineDataSet
        if (dataSet == null) {
            dataSet = LineDataSet(mutableListOf(), label).apply {
                this.color = color
                setDrawCircles(false)
                lineWidth = 1.8f
                setDrawValues(false)
                mode = LineDataSet.Mode.CUBIC_BEZIER
            }
            lineData.addDataSet(dataSet)
        }

        dataSet.addEntry(Entry(xVal, yVal))
        lineData.notifyDataChanged()
        binding.bvpChart.notifyDataSetChanged()
        binding.bvpChart.invalidate()
    }

    private fun onScanComplete() {
        runOnUiThread {
            cameraService?.unlockExposureAndWhiteBalance()
            executeInference()
        }
    }

    private fun executeInference() {
        appState = AppState.PROCESSING
        binding.progressBar.visibility = View.GONE
        binding.loadingSpinner.visibility = View.VISIBLE
        binding.btnStartScan.text = "PROCESSING..."
        binding.tvDiagnosisBadge.text = "PROCESSING RAW FRAMES & SOLVING HRVP-PINN PHYSICS..."

        coroutineScope.launch(Dispatchers.Default) {
            try {
                val py = Python.getInstance()
                val module = py.getModule("dual_site_engine")

                val modeStr = when (scanMode) {
                    ScanMode.DUAL_SITE_SIMULTANEOUS -> "dual_site"
                    ScanMode.FACE_ONLY -> "face_only"
                    ScanMode.FINGER_ONLY -> "finger_only"
                }

                val heightM = userHeightCm / 100.0f
                val deltaHMeters = if (scanMode == ScanMode.FINGER_ONLY) 0.0f else (0.18f * heightM)

                val resultJsonStr = module.callAttr(
                    "process_dual_site_stream",
                    faceRgbBuffer.toTypedArray(),
                    faceTsBuffer.toTypedArray(),
                    fingerRgbBuffer.toTypedArray(),
                    fingerTsBuffer.toTypedArray(),
                    imuGravityZBuffer.toTypedArray(),
                    deltaHMeters,
                    heightM,
                    userAge.toFloat(),
                    activeModelTier,
                    modeStr
                ).toString()

                val result = JSONObject(resultJsonStr)

                // High-Performance Native PulseFormer-Lite-50M ONNX Inference
                val isValid = result.optBoolean("is_valid", false)
                val faceWaveJson = result.optJSONArray("face_wave_7500")
                val palmWaveJson = result.optJSONArray("palm_wave_7500")
                val physioJson = result.optJSONArray("physio_14d")

                if (isValid && faceWaveJson != null && palmWaveJson != null && physioJson != null && pulseFormerEngine.isLoaded) {
                    try {
                        val faceArr = FloatArray(7500) { faceWaveJson.getDouble(it).toFloat() }
                        val palmArr = FloatArray(7500) { palmWaveJson.getDouble(it).toFloat() }
                        val physioArr = FloatArray(14) { physioJson.getDouble(it).toFloat() }

                        val pfOutput = pulseFormerEngine.runInference(faceArr, palmArr, physioArr)
                        if (pfOutput != null) {
                            result.put("sbp_pf", pfOutput.sbp.toDouble())
                            result.put("dbp_pf", pfOutput.dbp.toDouble())
                            result.put("map_pf", pfOutput.map.toDouble())
                            result.put("pf_infer_ms", pfOutput.inferenceMs.toDouble())

                            if (activeModelTier == "pf_50m") {
                                // Direct Foundation Student Inference: Output pure PulseFormer predictions directly
                                val sbpFinal = pfOutput.sbp.coerceIn(70f, 210f)
                                val dbpFinal = pfOutput.dbp.coerceIn(45f, 130f)
                                val mapFinal = if (pfOutput.map > 50f) pfOutput.map else (dbpFinal + (sbpFinal - dbpFinal) / 3.0f)

                                result.put("sbp", sbpFinal.toDouble())
                                result.put("dbp", dbpFinal.toDouble())
                                result.put("map", mapFinal.toDouble())
                            }
                        }
                    } catch (e: Throwable) {
                        Log.w("MainActivity", "Native ONNX execution warning: ${e.message}")
                    }
                }

                lastProcessedResultJson = result

                withContext(Dispatchers.Main) {
                    renderInferenceResults(result)
                }

            } catch (e: Throwable) {
                withContext(Dispatchers.Main) {
                    binding.loadingSpinner.visibility = View.GONE
                    binding.btnStartScan.text = "START SCAN"
                    binding.tvDiagnosisBadge.text = "INFERENCE ERROR: ${e.message}"
                    appState = AppState.IDLE
                    Toast.makeText(this@MainActivity, "Inference Failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    @SuppressLint("SetTextI18n")
    private fun renderInferenceResults(res: JSONObject) {
        binding.loadingSpinner.visibility = View.GONE
        binding.btnStartScan.text = "START NEW SCAN"
        appState = AppState.RESULT

        val isValid = res.optBoolean("is_valid", false)
        val message = res.optString("message", "")

        if (!isValid) {
            binding.tvDiagnosisBadge.text = "SCAN REJECTED: $message"
            binding.tvDiagnosisBadge.setBackgroundColor(Color.parseColor("#B91C1C"))
            return
        }

        binding.tvDiagnosisBadge.text = "SUCCESS: $message"
        binding.tvDiagnosisBadge.setBackgroundColor(Color.parseColor("#047857"))

        // Render Clean Phase-Aligned Overlapping Pulse Waves on the Chart
        val rppgWave = res.optJSONArray("rppg_wave")
        val palmWave = res.optJSONArray("finger_wave")
        if (rppgWave != null && palmWave != null && rppgWave.length() > 0 && palmWave.length() > 0) {
            val lineData = LineData()
            val faceEntries = mutableListOf<Entry>()
            val palmEntries = mutableListOf<Entry>()

            val nPts = minOf(rppgWave.length(), palmWave.length())
            for (i in 0 until nPts) {
                faceEntries.add(Entry(i.toFloat(), rppgWave.getDouble(i).toFloat()))
                palmEntries.add(Entry(i.toFloat(), palmWave.getDouble(i).toFloat()))
            }

            val faceSet = LineDataSet(faceEntries, "Site 1: Face rPPG (Carotid)").apply {
                color = Color.parseColor("#38BDF8") // Cyan
                setDrawCircles(false)
                lineWidth = 2.4f
                setDrawValues(false)
                mode = LineDataSet.Mode.CUBIC_BEZIER
            }

            val label2 = if (scanMode == ScanMode.DUAL_SITE_SIMULTANEOUS) {
                "Site 2: Palm rPPG (Phase-Aligned Overlap)"
            } else {
                "Site 2: Contact PPG"
            }

            val palmSet = LineDataSet(palmEntries, label2).apply {
                color = Color.parseColor("#F59E0B") // Amber
                setDrawCircles(false)
                lineWidth = 2.4f
                setDrawValues(false)
                mode = LineDataSet.Mode.CUBIC_BEZIER
            }

            lineData.addDataSet(faceSet)
            lineData.addDataSet(palmSet)
            binding.bvpChart.data = lineData
            binding.bvpChart.notifyDataSetChanged()
            binding.bvpChart.invalidate()
        }

        val sbp = res.optDouble("sbp", 0.0)
        val dbp = res.optDouble("dbp", 0.0)
        val map = res.optDouble("map", 0.0)
        binding.tvBP.text = "${sbp.toInt()} / ${dbp.toInt()}"
        binding.tvMAP.text = "MAP: ${String.format(Locale.US, "%.1f", map)} mmHg"

        val hr = res.optDouble("hr_bpm", 0.0)
        val hrv = res.optDouble("hrv_rmssd_ms", 0.0)
        val rr = res.optDouble("rr_brpm", 0.0)
        binding.tvHR.text = "${hr.toInt()} BPM"
        binding.tvHRV.text = "HRV: ${String.format(Locale.US, "%.1f", hrv)} ms | Resp: ${rr.toInt()}/m"

        val ptt = res.optDouble("delta_ptt_ms", 0.0)
        val povi = res.optDouble("povi_match_pct", 95.0)
        val epwv = res.optDouble("epwv_mps", 0.0)
        val pathM = String.format(Locale.US, "%.2f", 0.6736 * (userHeightCm / 100.0))
        val headDh = res.optDouble("delta_h_cm", 0.0)
        binding.tvHemodynamics.text =
            "PTT: ${String.format(Locale.US, "%.1f", ptt)} ms | POVI: ${String.format(Locale.US, "%.1f", povi)}% | ePWV: ${String.format(Locale.US, "%.2f", epwv)} m/s | Path: ${pathM}m | Head Δh: +${String.format(Locale.US, "%.1f", headDh)}cm"

        val sqiFace = res.optDouble("sqi_face_db", 0.0)
        val sqiFinger = res.optDouble("sqi_finger_db", 0.0)
        val procMs = res.optDouble("signal_proc_ms", 0.0)
        val inferMs = res.optDouble("model_infer_ms", 0.0)
        binding.tvPipelineAudit.text =
            "INPUT: SQI Face ${String.format(Locale.US, "%.1f", sqiFace)} dB, Hand ${String.format(Locale.US, "%.1f", sqiFinger)} dB | Proc: ${procMs.toInt()}ms | Infer: ${inferMs.toInt()}ms"

        val sbpPf = res.optDouble("sbp_pf", 0.0)
        val dbpPf = res.optDouble("dbp_pf", 0.0)
        val pfInferMs = res.optDouble("pf_infer_ms", 0.0)
        val sbpXl = res.optDouble("sbp_xl", 0.0)
        val dbpXl = res.optDouble("dbp_xl", 0.0)
        val sbpCompact = res.optDouble("sbp_compact", 0.0)
        val dbpCompact = res.optDouble("dbp_compact", 0.0)
        val sbpAna = res.optDouble("sbp_analytical", 0.0)
        val dbpAna = res.optDouble("dbp_analytical", 0.0)

        if (sbpPf > 0.0) {
            binding.tvModelBreakdown.text =
                "PF-50M (${pfInferMs.toInt()}ms): ${sbpPf.toInt()}/${dbpPf.toInt()} | TCN-XL: ${sbpXl.toInt()}/${dbpXl.toInt()} | HRVP: ${sbpAna.toInt()}/${dbpAna.toInt()} mmHg"
        } else {
            binding.tvModelBreakdown.text =
                "TCN-XL: ${sbpXl.toInt()}/${dbpXl.toInt()} | Compact: ${sbpCompact.toInt()}/${dbpCompact.toInt()} | HRVP: ${sbpAna.toInt()}/${dbpAna.toInt()} mmHg"
        }
    }

    private fun exportSessionCsv() {
        val result = lastProcessedResultJson
        if (result == null) {
            Toast.makeText(this, "No completed scan data available to export.", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val file = File(downloadsDir, "pulsefusion_scan_$timeStamp.csv")

            FileWriter(file).use { writer ->
                writer.append("Timestamp,Mode,Model,Height_cm,Age,SBP,DBP,MAP,PF_SBP,PF_DBP,HR_BPM,HRV_ms,PTT_ms,POVI_pct,ePWV_mps,SQI_Face_dB,SQI_Hand_dB\n")
                writer.append("$timeStamp,${scanMode.name},$activeModelTier,$userHeightCm,$userAge,")
                writer.append("${result.optDouble("sbp")},${result.optDouble("dbp")},${result.optDouble("map")},")
                writer.append("${result.optDouble("sbp_pf", result.optDouble("sbp"))},${result.optDouble("dbp_pf", result.optDouble("dbp"))},")
                writer.append("${result.optDouble("hr_bpm")},${result.optDouble("hrv_rmssd_ms")},${result.optDouble("delta_ptt_ms")},")
                writer.append("${result.optDouble("povi_match_pct", 95.0)},${result.optDouble("epwv_mps")},")
                writer.append("${result.optDouble("sqi_face_db")},${result.optDouble("sqi_finger_db")}\n")
            }

            Toast.makeText(this, "Exported to Downloads: ${file.name}", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to export CSV: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            latestGravityX = it.values[0]
            latestGravityY = it.values[1]
            latestGravityZ = it.values[2]
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
