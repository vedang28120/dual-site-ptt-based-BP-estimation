package com.example.pulsefusionbp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    enum class OverlayMode {
        DUAL_SITE,
        FACE_ONLY,
        FINGER_ONLY
    }

    private val roiPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#38BDF8")
        style = Paint.Style.STROKE
        strokeWidth = 3.5f
    }

    private val roiFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1538BDF8")
        style = Paint.Style.FILL
    }

    private val roiLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#38BDF8")
        textSize = 17f
        isFakeBoldText = true
    }

    private val meshPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#9900E5FF")
        style = Paint.Style.FILL
        strokeWidth = 3.5f
    }

    private val palmActivePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00E5FF")
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private val palmActiveFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#2000E5FF")
        style = Paint.Style.FILL
    }

    private val palmWaitingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F59E0B")
        style = Paint.Style.STROKE
        strokeWidth = 4f
        pathEffect = DashPathEffect(floatArrayOf(16f, 10f), 0f)
    }

    private val rearContactOkPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00E5FF")
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private val rearContactWaitPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#888888")
        style = Paint.Style.STROKE
        strokeWidth = 4f
        pathEffect = DashPathEffect(floatArrayOf(16f, 10f), 0f)
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 21f
        isFakeBoldText = true
    }

    private val labelActivePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00E5FF")
        textSize = 21f
        isFakeBoldText = true
    }

    private val labelWaitPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F59E0B")
        textSize = 21f
        isFakeBoldText = true
    }

    private var currentMode = OverlayMode.DUAL_SITE
    private var foreheadBox: RectF? = null
    private var leftCheekBox: RectF? = null
    private var rightCheekBox: RectF? = null
    private var facePoints: FloatArray? = null
    private var isPalmDetected: Boolean = false
    private var rearContactValid: Boolean = false

    // PERFECT SQUARE Palm guide target box in normalized square coordinates [0..1] x [0..1]
    // Center-Right alignment: Width = 0.28, Height = 0.28, Center Y = 0.50 -> [0.68..0.96, 0.36..0.64]
    val palmSquareRect = RectF(0.68f, 0.36f, 0.96f, 0.64f)

    fun setOverlayMode(mode: OverlayMode) {
        this.currentMode = mode
        postInvalidate()
    }

    fun updateFaceAndPalm(
        forehead: RectF?,
        leftCheek: RectF?,
        rightCheek: RectF?,
        points: FloatArray?,
        palmDetected: Boolean
    ) {
        this.foreheadBox = forehead
        this.leftCheekBox = leftCheek
        this.rightCheekBox = rightCheek
        this.facePoints = points
        this.isPalmDetected = palmDetected
        postInvalidate()
    }

    fun updateRearCropOverlay(contactValid: Boolean) {
        this.currentMode = OverlayMode.FINGER_ONLY
        this.rearContactValid = contactValid
        this.foreheadBox = null
        this.leftCheekBox = null
        this.rightCheekBox = null
        this.facePoints = null
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        if (viewWidth <= 0f || viewHeight <= 0f) return

        if (currentMode == OverlayMode.FINGER_ONLY) {
            // Square 1:1 center crop on rear contact
            val left = viewWidth * 0.35f
            val right = viewWidth * 0.65f
            val top = viewHeight * 0.35f
            val bottom = viewHeight * 0.65f
            val paint = if (rearContactValid) rearContactOkPaint else rearContactWaitPaint
            canvas.drawRect(left, top, right, bottom, paint)
            val msg = if (rearContactValid) "REAR CONTACT: OK" else "PLACE FINGER OVER LENS & FLASH"
            val textPaint = if (rearContactValid) labelActivePaint else labelPaint
            canvas.drawText(msg, (viewWidth - textPaint.measureText(msg)) / 2f, top - 16f, textPaint)
            return
        }

        // Draw Face Mesh points in square coordinates
        facePoints?.let { pts ->
            val scaledPts = FloatArray(pts.size)
            for (i in pts.indices step 2) {
                // x in [0..1], y mapped from portrait [0.125..0.875] to square [0..1]
                val xSq = pts[i]
                val ySq = (pts[i + 1] - 0.125f) / 0.75f
                scaledPts[i] = xSq * viewWidth
                scaledPts[i + 1] = ySq * viewHeight
            }
            canvas.drawPoints(scaledPts, meshPaint)
        }

        // Draw Horizontal Forehead ROI Band
        foreheadBox?.let {
            val left = it.left * viewWidth
            val right = it.right * viewWidth
            val top = ((it.top - 0.125f) / 0.75f) * viewHeight
            val bottom = ((it.bottom - 0.125f) / 0.75f) * viewHeight
            canvas.drawRoundRect(left, top, right, bottom, 10f, 10f, roiFillPaint)
            canvas.drawRoundRect(left, top, right, bottom, 10f, 10f, roiPaint)
        }

        // Draw Left Cheekbone Extended ROI
        leftCheekBox?.let {
            val left = it.left * viewWidth
            val right = it.right * viewWidth
            val top = ((it.top - 0.125f) / 0.75f) * viewHeight
            val bottom = ((it.bottom - 0.125f) / 0.75f) * viewHeight
            canvas.drawRoundRect(left, top, right, bottom, 10f, 10f, roiFillPaint)
            canvas.drawRoundRect(left, top, right, bottom, 10f, 10f, roiPaint)
        }

        // Draw Right Cheekbone Extended ROI
        rightCheekBox?.let {
            val left = it.left * viewWidth
            val right = it.right * viewWidth
            val top = ((it.top - 0.125f) / 0.75f) * viewHeight
            val bottom = ((it.bottom - 0.125f) / 0.75f) * viewHeight
            canvas.drawRoundRect(left, top, right, bottom, 10f, 10f, roiFillPaint)
            canvas.drawRoundRect(left, top, right, bottom, 10f, 10f, roiPaint)
        }

        // If in Dual-Site Mode, draw dedicated SQUARE Palm Guide Box at the side of the screen
        if (currentMode == OverlayMode.DUAL_SITE) {
            val pLeft = palmSquareRect.left * viewWidth
            val pTop = palmSquareRect.top * viewHeight
            val pRight = palmSquareRect.right * viewWidth
            val pBottom = palmSquareRect.bottom * viewHeight

            if (isPalmDetected) {
                canvas.drawRect(pLeft, pTop, pRight, pBottom, palmActiveFillPaint)
                canvas.drawRect(pLeft, pTop, pRight, pBottom, palmActivePaint)
                canvas.drawText("PALM (SITE 2): OK", pLeft, pTop - 10f, labelActivePaint)
            } else {
                canvas.drawRect(pLeft, pTop, pRight, pBottom, palmWaitingPaint)
                canvas.drawText("HOLD PALM HERE", pLeft, pTop - 10f, labelWaitPaint)
            }
        }
    }
}
