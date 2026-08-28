package com.example.activeperception

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.view.InputDevice
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.GridLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.activeperception.acquire.Detection
import com.example.activeperception.acquire.Grid
import com.example.activeperception.acquire.RAYNEO_X3_PRO_3x3
import com.example.activeperception.acquire.TfliteYoloDetector
import org.json.JSONObject
import java.io.File
import kotlin.math.abs
import kotlin.math.max

/**
 * UI shell: pick a mode and its parameters, then run it on a worker thread. All measurement
 * behaviour lives in [MeasurementController] and the output layout in [MeasurementLogger].
 *
 * The camera and detector are opened once per Activity session, but a fresh logger and
 * controller are built per run so each lands in its own directory.
 */
class MeasurementActivity : AppCompatActivity() {

    companion object {
        private const val RAYNEO_FRAME_PERIOD_NS = 33_329_000L
        private const val START_DISPATCH_ALLOWANCE_NS = 10_000_000L
    }

    private val grid: Grid = RAYNEO_X3_PRO_3x3
    private lateinit var raw: RawSensorCapturer
    private lateinit var detector: TfliteYoloDetector
    private lateinit var sensors: SensorDataManager
    private lateinit var health: DeviceHealthMonitor
    private lateinit var status: TextView
    private lateinit var preview: ImageView
    private lateinit var overlay: OverlayView
    private lateinit var cellText: TextView
    private lateinit var methodGroup: RadioGroup
    private lateinit var cellGridWrap: LinearLayout
    private lateinit var cellGrid: GridLayout
    private lateinit var periodGroup: RadioGroup
    private lateinit var fallbackGroup: RadioGroup
    private lateinit var boostGroup: RadioGroup
    private lateinit var proposedSettings: LinearLayout
    private lateinit var aeSettings: LinearLayout
    private lateinit var aeStrategyGroup: RadioGroup
    private lateinit var confThreshSpinner: Spinner
    private lateinit var offloadCheck: CheckBox
    private lateinit var offloadUrl: EditText
    private lateinit var offloadRegime: Spinner
    private lateinit var controlsScroll: ScrollView
    private lateinit var controlsInner: LinearLayout
    private lateinit var btnStart: Button
    private lateinit var btnRotationStart: Button
    private lateinit var btnStop: Button
    private val cellButtons = ArrayList<Button>(16)
    private var selectedCell: Int = 0

    // The X3 Pro temple pad reports absolute touchscreen coordinates that are unrelated to
    // the UI. Treat it as a focus controller, while phones retain normal touch behaviour.
    private val rayNeoTouchpad = Build.MANUFACTURER.equals("RayNeo", ignoreCase = true)
    private var pointerDownX = 0f
    private var pointerDownY = 0f
    private var pointerSwipeHandled = false

    private var mc: MeasurementController? = null
    // TFLite GPU delegates must be created, invoked, and closed on the same thread. Reuse one
    // serial worker across Start presses instead of creating a new Thread for every run.
    private val inferenceExecutor = Executors.newSingleThreadExecutor { task ->
        Thread(task, "SoS-GPU-Worker")
    }
    private val displayIoExecutor = Executors.newSingleThreadExecutor { task ->
        Thread(task, "SoS-DisplayIo").apply { priority = Thread.MIN_PRIORITY }
    }
    @Volatile private var runActive = false
    private var opened = false
    @Volatile private var resourcesMustClose = false
    @Volatile private var displayConfThreshold = 0.25f

    private class RotationRunSession(val id: Long) {
        @Volatile var cancelled = false
        @Volatile var measurementStarted = false
        @Volatile var logger: MeasurementLogger? = null
        @Volatile var trigger: RotationStartController? = null
        val cleanupClaimed = AtomicBoolean(false)
        val firstPoseRecorded = AtomicBoolean(false)
    }
    @Volatile private var rotationSession: RotationRunSession? = null

    /** Non-null when the Offload checkbox is set, or when the Activity was launched with an
     *  `--es server_url http://...` extra. */
    private var offloader: OffloadClient? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // A screen-off event pauses the Activity on RayNeo. Keep the display awake during an
        // experiment; closing TFLite while Interpreter.run() is active causes a native crash.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_measurement)
        status = findViewById(R.id.statusText)
        preview = findViewById(R.id.previewImage)
        overlay = findViewById(R.id.overlayView)
        cellText = findViewById(R.id.cellText)
        methodGroup = findViewById(R.id.methodGroup)
        cellGridWrap = findViewById(R.id.cellGridWrap)
        cellGrid = findViewById(R.id.cellGrid)
        periodGroup = findViewById(R.id.periodGroup)
        fallbackGroup = findViewById(R.id.fallbackGroup)
        boostGroup = findViewById(R.id.boostGroup)
        proposedSettings = findViewById(R.id.proposedSettings)
        aeSettings = findViewById(R.id.aeSettings)
        aeStrategyGroup = findViewById(R.id.aeStrategyGroup)
        confThreshSpinner = findViewById(R.id.confThreshSpinner)
        val confChoices = (1..10).map { "%.2f".format(it * 0.05) }   // 0.05 .. 0.50
        confThreshSpinner.adapter = ArrayAdapter(this,
            android.R.layout.simple_spinner_item, confChoices
        ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
        confThreshSpinner.setSelection(confChoices.indexOf("0.25").coerceAtLeast(0))
        offloadCheck = findViewById(R.id.offloadCheck)
        offloadUrl = findViewById(R.id.offloadUrl)
        offloadRegime = findViewById(R.id.offloadRegime)
        controlsScroll = findViewById(R.id.controls)
        controlsInner = findViewById(R.id.controlsInner)
        btnStart = findViewById(R.id.btnStart)
        btnRotationStart = findViewById(R.id.btnRotationStart)
        btnStop = findViewById(R.id.btnStop)
        // Server-side delay-injection profiles; "clear" means no added delay. New labels must
        // be added to the server's regime table too.
        offloadRegime.adapter = ArrayAdapter(this,
            android.R.layout.simple_spinner_item,
            listOf("clear", "wifi", "5g", "lte", "congested")
        ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
        sensors = SensorDataManager(this)
        health = DeviceHealthMonitor(this)

        // Edge-to-edge on Android 15+. All four sides, because the nav bar moves to the right
        // edge in landscape on some devices. Base padding is captured once so repeated
        // callbacks don't compound it.
        val basePad = intArrayOf(
            controlsInner.paddingLeft, controlsInner.paddingTop,
            controlsInner.paddingRight, controlsInner.paddingBottom)
        ViewCompat.setOnApplyWindowInsetsListener(controlsInner) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(
                basePad[0] + bars.left,
                basePad[1] + bars.top,
                basePad[2] + bars.right,
                basePad[3] + bars.bottom)
            insets
        }

        applyOrientation(resources.configuration.orientation)
        // Must precede buildCellGrid so the cell labels render with the right effective ISO.
        grid.digitalBoost = boostFromCheckedId(boostGroup.checkedRadioButtonId)
        buildCellGrid()
        methodGroup.setOnCheckedChangeListener { _, id ->
            cellGridWrap.visibility = if (id == R.id.methodFixed) View.VISIBLE else View.GONE
            proposedSettings.visibility = if (id == R.id.methodProposed) View.VISIBLE else View.GONE
            aeSettings.visibility =
                if (id == R.id.methodAe || id == R.id.methodAeQuant) View.VISIBLE else View.GONE
        }
        // Initial state mirrors the default-checked methodProposed radio.
        cellGridWrap.visibility = View.GONE
        proposedSettings.visibility = View.VISIBLE
        aeSettings.visibility = View.GONE

        // Grid is mutated in place; the rebuild refreshes the effective-ISO labels.
        boostGroup.setOnCheckedChangeListener { _, id ->
            grid.digitalBoost = boostFromCheckedId(id)
            buildCellGrid()
        }

        btnStart.setOnClickListener { startSelectedMethod() }
        btnRotationStart.setOnClickListener { startRotationSelectedMethod() }
        findViewById<Button>(R.id.btnVerify).setOnClickListener {
            start("verify") { mc!!.runVerify(false, ::post, ::showFrame) }
        }
        findViewById<Button>(R.id.btnVerifyProbe).setOnClickListener {
            start("verifyprobe") { mc!!.runVerifyProbe(false, ::post, ::showFrame) }
        }
        findViewById<Button>(R.id.btnBench).setOnClickListener {
            start("ab_bench") { mc!!.runAbBench(::post) }
        }
        findViewById<Button>(R.id.btnIsoDiag).setOnClickListener {
            start("iso_diag") { mc!!.runIsoDiag(onStatus = ::post) }
        }
        findViewById<Button>(R.id.btnExp21).setOnClickListener {
            start("exp2_1") { mc!!.runExp21DirectTensor(::post) }
        }
        findViewById<Button>(R.id.btnExp22).setOnClickListener {
            start("exp2_2") { mc!!.runExp22DecodeOptimization(::post) }
        }
        findViewById<Button>(R.id.btnExp23).setOnClickListener {
            start("exp2_3") { mc!!.runExp23Coco5Comparison(::post) }
        }
        findViewById<Button>(R.id.btnExp3).setOnClickListener {
            start("exp3") { mc!!.runExp3IntegratedVsExp1A(::post) }
        }
        findViewById<Button>(R.id.btnExp4).setOnClickListener {
            start("exp4") { mc!!.runExp4PipelinedPeriods(::post) }
        }
        findViewById<Button>(R.id.btnExp51).setOnClickListener {
            start("exp5_1") { mc!!.runExp51CaptureGuardComparison(::post) }
        }
        btnStop.setOnClickListener { stopMeasurement() }

        configureStaticTouchpadControls()
        // Start is the safest useful default: one tap launches the default Proposed mode.
        if (rayNeoTouchpad) btnStart.post { btnStart.requestFocus() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        } else if (intent.getStringExtra("autorun") == "ab_visual") {
            // Headless-friendly entry point for a controlled paired A/B capture. Posting keeps
            // initialization ordered after the Activity has finished wiring its views.
            btnStart.post {
                start("ab_visual") { mc!!.runAbVisual(::post) }
            }
        } else if (intent.getStringExtra("autorun") == "exp2_1") {
            btnStart.post {
                start("exp2_1") { mc!!.runExp21DirectTensor(::post) }
            }
        } else if (intent.getStringExtra("autorun") == "exp2_2") {
            btnStart.post {
                start("exp2_2") { mc!!.runExp22DecodeOptimization(::post) }
            }
        } else if (intent.getStringExtra("autorun") == "exp2_3") {
            btnStart.post {
                start("exp2_3") { mc!!.runExp23Coco5Comparison(::post) }
            }
        } else if (intent.getStringExtra("autorun") == "exp3") {
            btnStart.post {
                start("exp3") { mc!!.runExp3IntegratedVsExp1A(::post) }
            }
        } else if (intent.getStringExtra("autorun") == "exp4") {
            btnStart.post {
                start("exp4") { mc!!.runExp4PipelinedPeriods(::post) }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_1") {
            btnStart.post {
                start("exp5_1") { mc!!.runExp51CaptureGuardComparison(::post) }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_1_a") {
            btnStart.post {
                start("exp5_1_a") { mc!!.runExp51CaptureGuardComparison(::post, strategyFilter = "A_") }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_1_b") {
            btnStart.post {
                start("exp5_1_b") { mc!!.runExp51CaptureGuardComparison(::post, strategyFilter = "B_") }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_1_c") {
            btnStart.post {
                start("exp5_1_c") { mc!!.runExp51CaptureGuardComparison(::post, strategyFilter = "C_") }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_1_1_k") {
            btnStart.post {
                start("exp5_1_1_k") { mc!!.runExp511NoGuardDecode(::post, "B_kotlin") }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_1_1_n") {
            btnStart.post {
                start("exp5_1_1_n") { mc!!.runExp511NoGuardDecode(::post, "D_native_neon") }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_1_1_p") {
            btnStart.post {
                start("exp5_1_1_p") { mc!!.runExp511NoGuardDecode(::post, "E_native_preview") }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_2_o") {
            btnStart.post {
                start("exp5_2_o") { mc!!.runExp52CaptureMode(::post, "A_on_demand") }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_2_c") {
            btnStart.post {
                start("exp5_2_c") { mc!!.runExp52CaptureMode(::post, "B_continuous_ring") }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_3") {
            btnStart.post {
                start("exp5_3") { mc!!.runExp53RawFormats(::post) }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_3_s") {
            btnStart.post {
                start("exp5_3_s") {
                    mc!!.runExp53RawFormats(::post, android.graphics.ImageFormat.RAW_SENSOR)
                }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_3_10") {
            btnStart.post {
                start("exp5_3_10") {
                    mc!!.runExp53RawFormats(::post, android.graphics.ImageFormat.RAW10)
                }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_4") {
            btnStart.post {
                start("exp5_4") { mc!!.runExp54CpuContention(::post) }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_4_best") {
            btnStart.post {
                start("exp5_4_best") {
                    mc!!.runExp54CpuContention(::post, confirmationOnly = true)
                }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_5") {
            btnStart.post {
                start("exp5_5") { mc!!.runExp55FinalAdaptiveP5(::post) }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_6") {
            btnStart.post {
                start("exp5_6") {
                    mc!!.runExp55FinalAdaptiveP5(::post, persistentFastCapture = true)
                }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_6_d2") {
            btnStart.post {
                start("exp5_6_d2") {
                    mc!!.runExp55FinalAdaptiveP5(::post, persistentFastCapture = true,
                        decodeThreads = 2, formationThreads = 4)
                }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_6_d3") {
            btnStart.post {
                start("exp5_6_d3") {
                    mc!!.runExp55FinalAdaptiveP5(::post, persistentFastCapture = true,
                        decodeThreads = 3, formationThreads = 4)
                }
            }
        } else if (intent.getStringExtra("autorun") == "exp5_7") {
            btnStart.post {
                start("exp5_7") {
                    mc!!.runExp55FinalAdaptiveP5(::post, persistentFastCapture = true,
                        decodeThreads = 4, formationThreads = 4,
                        deepSinglePrefetch = true)
                }
            }
        } else if (intent.getStringExtra("autorun") == "exp6") {
            btnStart.post {
                start("exp6") {
                    mc!!.runExp55FinalAdaptiveP5(::post, persistentFastCapture = true,
                        decodeThreads = 4, formationThreads = 4,
                        deepSinglePrefetch = true, integratedOptimizations = true)
                }
            }
        }

    }

    /** Reflows the root between portrait (preview above controls) and landscape (controls
     *  left, preview right) by swapping LayoutParams rather than reinflating — reinflating
     *  would tear down the camera and detector. View objects keep their state across the
     *  detach and reattach. */
    private fun applyOrientation(orientation: Int) {
        val root = findViewById<LinearLayout>(R.id.rootLayout)
        // Grab the children before removeAllViews; afterwards findViewById returns null.
        val previewWrap = findViewById<android.widget.FrameLayout>(R.id.previewWrap)
        val controls = findViewById<android.widget.ScrollView>(R.id.controls)
        root.removeAllViews()
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
            root.orientation = LinearLayout.HORIZONTAL
            root.addView(controls, LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.MATCH_PARENT, 2f))
            root.addView(previewWrap, LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.MATCH_PARENT, 3f))
        } else {
            root.orientation = LinearLayout.VERTICAL
            root.addView(previewWrap, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
            root.addView(controls, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        applyOrientation(newConfig.orientation)
        // System bars move on rotation, so re-fire the inset listener to follow them.
        ViewCompat.requestApplyInsets(findViewById(R.id.controlsInner))
    }

    private fun boostFromCheckedId(id: Int): Double = when (id) {
        R.id.boost1x -> 1.0
        R.id.boost4x -> 4.0
        else -> 2.0
    }

    private fun aeStrategyFromCheckedId(id: Int): AeStrategy = when (id) {
        R.id.aeCustom -> AeStrategy.CUSTOM_BRIGHTNESS
        else -> AeStrategy.PHONE
    }

    private fun effectiveIso(gainIdx: Int): Int =
        (grid.gains[gainIdx] * grid.digitalBoost).toInt()

    /** Fixed-mode cell selector: rows are gain, columns are shutter. Labels show EFFECTIVE
     *  ISO so the number matches the brightness the formed cell will actually have. Call
     *  again after digitalBoost changes to refresh them. */
    private fun buildCellGrid() {
        cellGrid.removeAllViews()
        cellButtons.clear()
        // Overrides the XML's 3×3 hint so wider grids get enough rows and columns.
        cellGrid.rowCount = grid.nGain
        cellGrid.columnCount = grid.nShutter
        for (gi in 0 until grid.nGain) {
            for (sj in 0 until grid.nShutter) {
                val cell = grid.cell(gi, sj)
                val effIso = effectiveIso(gi); val expUs = grid.exposuresUs[sj]
                // Two lines, no unit words — anything longer stops fitting on a phone.
                val expMs = expUs / 1000.0
                val expLabel = if (expMs >= 1.0) "%.0fms".format(expMs) else "%.1fms".format(expMs)
                val b = Button(this).apply {
                    text = "$effIso\n$expLabel"
                    textSize = 10f
                    minWidth = 0; minimumWidth = 0
                    setPadding(0, 0, 0, 0)
                    setOnClickListener { setSelectedCell(cell) }
                }
                configureTouchpadControl(b)
                val lp = GridLayout.LayoutParams().apply {
                    width = 0; height = GridLayout.LayoutParams.WRAP_CONTENT
                    columnSpec = GridLayout.spec(sj, 1f); rowSpec = GridLayout.spec(gi, 1f)
                    setMargins(2, 2, 2, 2)
                }
                cellGrid.addView(b, lp)
                cellButtons.add(b)
            }
        }
        setSelectedCell(0)
    }

    private fun setSelectedCell(cell: Int) {
        selectedCell = cell
        val idle = ContextCompat.getColor(this, R.color.cell_unselected)
        val active = ContextCompat.getColor(this, R.color.cell_selected)
        for (b in cellButtons) b.setBackgroundColor(idle)
        cellButtons.getOrNull(cell)?.setBackgroundColor(active)
    }

    private fun startSelectedMethod() {
        // GT-reference checkbox removed from UI; hardcoded false. Re-add a flag plumb
        // if a GT pass mode is needed (or a long-press gesture on Start to toggle).
        val gtRef = false
        when (methodGroup.checkedRadioButtonId) {
            R.id.methodFixed -> {
                val (gi, sj) = grid.indices(selectedCell)
                start("fixed_g${grid.gains[gi]}_e${grid.exposuresUs[sj]}") {
                    mc!!.runFixed(gi, sj, 300, gtRef, ::post, ::onFrameWithOffload)
                }
            }
            R.id.methodAe -> {
                val ae = aeStrategyFromCheckedId(aeStrategyGroup.checkedRadioButtonId)
                start("ae_${ae.tag()}") { mc!!.runAe(300, gtRef, ae, ::post, ::onFrameWithOffload) }
            }
            R.id.methodAeQuant -> {
                val ae = aeStrategyFromCheckedId(aeStrategyGroup.checkedRadioButtonId)
                start("ae_paired_${ae.tag()}") { mc!!.runAeQuant(300, gtRef, ae, ::post, ::onFrameWithOffload) }
            }
            R.id.methodProposed -> {
                val period = when (periodGroup.checkedRadioButtonId) {
                    R.id.period9 -> 9
                    R.id.period12 -> 12
                    else -> 5
                }
                val fallback = when (fallbackGroup.checkedRadioButtonId) {
                    R.id.fbLaplacian -> FallbackMetric.LAPLACIAN_VAR
                    R.id.fbTenengrad -> FallbackMetric.TENENGRAD_NORM
                    R.id.fbCreteRoffet -> FallbackMetric.CRETE_ROFFET
                    R.id.fbSafeCell -> FallbackMetric.SAFE_CELL
                    else -> FallbackMetric.ENTROPY
                }
                start("proposed_p${period}_${fallback.tag()}") {
                    mc!!.runFinalProposed(period, 300, fallback,
                        onStatus = ::post, onFrame = ::onFrameWithOffload)
                }
            }
            else -> post("pick a method")
        }
    }

    /** The ordinary Start path above is intentionally unchanged. This separate path snapshots
     *  the selected method, preloads camera/GPU, learns the oscillation, then releases exactly
     *  one run at the learned midpoint while moving in +gyro-Y. */
    private fun startRotationSelectedMethod() {
        val gtRef = false
        when (methodGroup.checkedRadioButtonId) {
            R.id.methodFixed -> {
                val (gi, sj) = grid.indices(selectedCell)
                val expUs = grid.exposuresUs[sj]
                startRotation("fixed_g${grid.gains[gi]}_e$expUs", expUs, grid.gains[gi], 1,
                    expUs * 1_000L / 2L) {
                    mc!!.runFixed(gi, sj, 300, gtRef, ::post, ::onFrameWithOffload)
                }
            }
            R.id.methodAe -> {
                val ae = aeStrategyFromCheckedId(aeStrategyGroup.checkedRadioButtonId)
                startRotation("ae_${ae.tag()}", grid.fastestExposureUs, grid.baseGain, 1,
                    grid.fastestExposureUs * 1_000L / 2L) {
                    mc!!.runAe(300, gtRef, ae, ::post, ::onFrameWithOffload)
                }
            }
            R.id.methodAeQuant -> {
                val ae = aeStrategyFromCheckedId(aeStrategyGroup.checkedRadioButtonId)
                startRotation("ae_paired_${ae.tag()}", grid.fastestExposureUs, grid.baseGain, 1,
                    grid.fastestExposureUs * 1_000L / 2L) {
                    mc!!.runAeQuant(300, gtRef, ae, ::post, ::onFrameWithOffload)
                }
            }
            R.id.methodProposed -> {
                val period = when (periodGroup.checkedRadioButtonId) {
                    R.id.period9 -> 9
                    R.id.period12 -> 12
                    else -> 5
                }
                val fallback = when (fallbackGroup.checkedRadioButtonId) {
                    R.id.fbLaplacian -> FallbackMetric.LAPLACIAN_VAR
                    R.id.fbTenengrad -> FallbackMetric.TENENGRAD_NORM
                    R.id.fbCreteRoffet -> FallbackMetric.CRETE_ROFFET
                    R.id.fbSafeCell -> FallbackMetric.SAFE_CELL
                    else -> FallbackMetric.ENTROPY
                }
                val halfWindowNs = ((grid.maxBurst - 1) * RAYNEO_FRAME_PERIOD_NS +
                    grid.fastestExposureUs * 1_000L) / 2L
                startRotation("proposed_p${period}_${fallback.tag()}",
                    grid.fastestExposureUs, grid.baseGain, grid.maxBurst, halfWindowNs) {
                    mc!!.runFinalProposed(period, 300, fallback,
                        onStatus = ::post, onFrame = ::onFrameWithOffload)
                }
            }
            else -> post("pick a method")
        }
    }

    private fun startRotation(
        modeTag: String,
        preflightExposureUs: Int,
        preflightIso: Int,
        preflightBurst: Int,
        firstWindowHalfNs: Long,
        block: () -> Unit
    ) {
        if (runActive) { post("busy — stop first"); return }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) { post("no camera permission"); return }

        // Snapshot every UI option before asynchronous preload/learning begins.
        val offUrl = (if (offloadCheck.isChecked) offloadUrl.text.toString().trim().ifBlank { null }
                      else null) ?: intent.getStringExtra("server_url")
        val offRegime = (offloadRegime.selectedItem?.toString() ?: "clear").ifBlank { "clear" }
        val selectConf = confThreshSpinner.selectedItem?.toString()?.toFloatOrNull() ?: 0.25f
        displayConfThreshold = selectConf
        val session = RotationRunSession(System.currentTimeMillis())
        rotationSession = session
        runActive = true

        inferenceExecutor.execute {
            try {
                if (resourcesMustClose) {
                    closeResources(); resourcesMustClose = false
                }
                if (session.cancelled) return@execute
                if (!opened) {
                    post("ROTATION PRELOAD · opening camera…")
                    raw = RawSensorCapturer(this); raw.open()
                    post("ROTATION PRELOAD · loading optimized COCO5 GPU batches…")
                    detector = TfliteYoloDetector(
                        this,
                        batchedAssets = listOf(
                            "yolov8n_640_coco5_fp16.tflite" to 1,
                            "yolov8n_640_b3_coco5_fp16.tflite" to 3
                        ),
                        numClasses = 5,
                        classIdMap = TfliteYoloDetector.COCO5_CLASS_IDS,
                        accelerator = TfliteYoloDetector.Accelerator.GPU,
                        allowFallback = false,
                        gpuBackend = TfliteYoloDetector.GpuBackend.AUTO,
                        onLoadStatus = ::post
                    )
                    opened = true
                }
                if (session.cancelled) return@execute

                post("ROTATION PRELOAD · warming GPU B=1/3…")
                detector.warmUpAllBatches()
                if (session.cancelled) return@execute

                // First pass settles the requested physical state. The second pass measures
                // request-to-first-sensor latency without the 12-frame setting guard.
                post("ROTATION PRELOAD · settling camera and AWB…")
                raw.capture(preflightExposureUs, preflightIso, preflightBurst)
                val submittedNs = SystemClock.elapsedRealtimeNanos()
                raw.capture(preflightExposureUs, preflightIso, preflightBurst)
                val firstSensorNs = raw.lastMeta.firstOrNull()?.timestamp ?: submittedNs
                val requestLatencyNs = (firstSensorNs - submittedNs).coerceAtLeast(0L)
                if (session.cancelled) return@execute

                val runName = "run_rotation_${modeTag}_${session.id}"
                val logger = MeasurementLogger(this, runName)
                session.logger = logger
                val mcLocal = MeasurementController(raw, detector, grid, sensors, logger, health, selectConf)
                mc = mcLocal
                offloader = offUrl?.let {
                    OffloadClient(it, logger.dir, offRegime) { mcLocal.lastFrameIdx }
                        .apply { warmConnection() }
                }

                val triggerController = RotationStartController(this, logger.dir, ::post)
                session.trigger = triggerController
                triggerController.startLearning { profile ->
                    if (session.cancelled || rotationSession !== session) return@startLearning
                    // Include worker dispatch/startPass allowance. The camera part is measured
                    // on this exact session and the exposure-window term is method-specific.
                    val leadNs = requestLatencyNs + firstWindowHalfNs + START_DISPATCH_ALLOWANCE_NS
                    post("ROTATION · ${"%.1f".format(profile.rangeDeg)}° learned · arming start")
                    triggerController.arm(leadNs) { trigger ->
                        if (session.cancelled || rotationSession !== session) return@arm
                        session.measurementStarted = true
                        File(logger.dir, "rotation_start.json").writeText(JSONObject().apply {
                            put("mode", modeTag)
                            put("range_deg", profile.rangeDeg)
                            put("half_period_ms", profile.halfPeriodMs)
                            put("direction", "gyro_y_positive")
                            put("preflight_exposure_us", preflightExposureUs)
                            put("preflight_iso", preflightIso)
                            put("preflight_burst", preflightBurst)
                            put("request_latency_ns", requestLatencyNs)
                            put("first_window_half_ns", firstWindowHalfNs)
                            put("dispatch_allowance_ns", START_DISPATCH_ALLOWANCE_NS)
                            put("trigger_sensor_timestamp_ns", trigger.sensorTimestampNs)
                            put("trigger_gyro_y_rad_s", trigger.gyroY)
                            put("trigger_center_distance_deg", trigger.centerDistanceDeg)
                            put("predicted_time_to_center_ns", trigger.predictedTimeToCenterNs)
                            put("lead_ns", trigger.leadNs)
                        }.toString(2))
                        post("ROTATION · trigger · measurement starting")
                        inferenceExecutor.execute {
                            if (session.cancelled) {
                                runCatching { triggerController.close() }
                                session.trigger = null
                                logger.close(); finishRotationSession(session)
                                return@execute
                            }
                            try {
                                block()
                                val above = mcLocal.detectionTotalAboveThresh
                                val floor = mcLocal.detectionTotalAtFloor
                                val frames = mcLocal.totalFramesLogged
                                post("rotation done — detections: $above above thresh ($floor at floor), $frames frames")
                            } catch (error: Throwable) {
                                post("rotation error: ${error.message}")
                            } finally {
                                runCatching { session.trigger?.close() }
                                session.trigger = null
                                logger.close()
                                finishRotationSession(session)
                            }
                        }
                    }
                }
            } catch (error: Throwable) {
                post("rotation preload error: ${error.message}")
                cleanupPendingRotationSession(session)
            } finally {
                if (session.cancelled && !session.measurementStarted) {
                    cleanupPendingRotationSession(session)
                }
            }
        }
    }

    private fun finishRotationSession(session: RotationRunSession) {
        if (rotationSession === session) rotationSession = null
        runActive = false
    }

    private fun cleanupPendingRotationSession(session: RotationRunSession) {
        if (!session.cleanupClaimed.compareAndSet(false, true)) return
        runCatching { session.trigger?.close() }
        session.trigger = null
        runCatching { session.logger?.close() }
        finishRotationSession(session)
    }

    private fun cancelRotationSession(): Boolean {
        val session = rotationSession ?: return false
        session.cancelled = true
        runCatching { session.trigger?.close() }
        session.trigger = null
        mc?.stop()
        post("ROTATION · cancelled")
        if (!session.measurementStarted) {
            runCatching { inferenceExecutor.execute { cleanupPendingRotationSession(session) } }
        }
        return true
    }

    private fun stopMeasurement() {
        if (cancelRotationSession()) {
            runOnUiThread { overlay.clear(); cellText.text = "—" }
            return
        }
        mc?.stop()
        post("stopping…")
        runOnUiThread { overlay.clear(); cellText.text = "—" }
    }

    private fun toggleRunFromTouchpad() {
        if (runActive) stopMeasurement() else startSelectedMethod()
    }

    /** DPAD-style input is emitted by some RayNeo firmware and is also useful for testing. */
    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        val direction = when (event.keyCode) {
            KeyEvent.KEYCODE_DPAD_DOWN, KeyEvent.KEYCODE_DPAD_RIGHT,
            KeyEvent.KEYCODE_TAB, KeyEvent.KEYCODE_PAGE_DOWN -> 1
            KeyEvent.KEYCODE_DPAD_UP, KeyEvent.KEYCODE_DPAD_LEFT,
            KeyEvent.KEYCODE_PAGE_UP -> -1
            else -> 0
        }
        if (direction != 0) {
            if (event.action == KeyEvent.ACTION_DOWN && event.repeatCount == 0) moveTouchpadFocus(direction)
            return true
        }
        if (event.keyCode in intArrayOf(
                KeyEvent.KEYCODE_DPAD_CENTER, KeyEvent.KEYCODE_ENTER, KeyEvent.KEYCODE_SPACE)) {
            if (event.action == KeyEvent.ACTION_UP) activateFocusedControl()
            return true
        }
        return super.dispatchKeyEvent(event)
    }

    /** Trackpad/mouse-wheel firmware variants map their scroll axis to the same focus model. */
    override fun dispatchGenericMotionEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_SCROLL &&
            event.isFromSource(InputDevice.SOURCE_CLASS_POINTER)) {
            val vertical = event.getAxisValue(MotionEvent.AXIS_VSCROLL)
            val horizontal = event.getAxisValue(MotionEvent.AXIS_HSCROLL)
            val axis = if (vertical != 0f) vertical else -horizontal
            if (axis != 0f) {
                moveTouchpadFocus(if (axis < 0f) 1 else -1)
                return true
            }
        }
        return super.dispatchGenericMotionEvent(event)
    }

    /**
     * RayNeo reports the temple pad as a direct absolute touchscreen. Coordinates therefore
     * cannot be used for hit-testing: swipes move focus and taps operate the focused control.
     */
    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        if (!rayNeoTouchpad) return super.dispatchTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                pointerDownX = event.x
                pointerDownY = event.y
                pointerSwipeHandled = false
            }
            MotionEvent.ACTION_MOVE -> if (!pointerSwipeHandled) {
                val dx = event.x - pointerDownX
                val dy = event.y - pointerDownY
                if (max(abs(dx), abs(dy)) >= dp(42)) {
                    pointerSwipeHandled = true
                    val forward = if (abs(dy) >= abs(dx)) dy < 0f else dx < 0f
                    moveTouchpadFocus(if (forward) 1 else -1)
                }
            }
            MotionEvent.ACTION_UP -> if (!pointerSwipeHandled) {
                if (event.eventTime - event.downTime >= ViewConfiguration.getLongPressTimeout()) {
                    toggleRunFromTouchpad()
                } else {
                    activateFocusedControl()
                }
            }
        }
        return true
    }

    private fun configureStaticTouchpadControls() {
        touchpadControls().forEach(::configureTouchpadControl)
    }

    /** Reads the current hierarchy so rebuilt Fixed cells and hidden method settings stay sane. */
    private fun touchpadControls(): List<View> {
        val result = ArrayList<View>()
        fun visit(view: View) {
            when (view) {
                is EditText -> Unit // Avoid opening an on-glass keyboard for the server URL.
                is Button, is CheckBox, is Spinner -> result += view
                is ViewGroup -> for (i in 0 until view.childCount) visit(view.getChildAt(i))
            }
        }
        visit(controlsInner)
        return result
    }

    private fun availableTouchpadControls(): List<View> =
        touchpadControls().filter { it.isShown && it.isEnabled }

    private fun currentTouchpadControl(): View? =
        currentFocus?.takeIf { it in touchpadControls() && it.isShown && it.isEnabled }

    private fun moveTouchpadFocus(delta: Int) {
        val available = availableTouchpadControls()
        if (available.isEmpty()) return
        val current = available.indexOf(currentTouchpadControl())
        val next = if (current < 0) 0 else (current + delta).floorMod(available.size)
        available[next].requestFocus()
        scrollControlIntoView(available[next])
    }

    private fun activateFocusedControl() {
        val focused = currentTouchpadControl() ?: btnStart.also { it.requestFocus() }
        if (focused is Spinner) {
            if (focused.count > 0) {
                focused.setSelection((focused.selectedItemPosition + 1).floorMod(focused.count))
                post("${controlLabel(focused)} · ${focused.selectedItem}")
            }
        } else {
            focused.performClick()
        }
    }

    private fun configureTouchpadControl(view: View) {
        view.isFocusable = true
        view.isFocusableInTouchMode = true
        view.setOnFocusChangeListener { focusedView, hasFocus ->
            focusedView.animate().cancel()
            focusedView.scaleX = if (hasFocus) 1.06f else 1f
            focusedView.scaleY = if (hasFocus) 1.06f else 1f
            focusedView.alpha = if (hasFocus) 1f else 0.9f
            if (hasFocus) {
                scrollControlIntoView(focusedView)
                if (!runActive) post("Focus · ${controlLabel(focusedView)} · tap")
            }
        }
    }

    private fun scrollControlIntoView(view: View) {
        controlsScroll.post {
            val rect = Rect()
            view.getDrawingRect(rect)
            controlsInner.offsetDescendantRectToMyCoords(view, rect)
            controlsScroll.smoothScrollTo(0, max(0, rect.top - dp(56)))
        }
    }

    private fun controlLabel(view: View): String = when (view) {
        is Spinner -> when (view.id) {
            R.id.confThreshSpinner -> "Confidence"
            R.id.offloadRegime -> "Network"
            else -> "Option"
        }
        is TextView -> view.text.toString().replace('\n', ' ').trim().ifBlank { "Control" }
        else -> "Control"
    }

    private fun Int.floorMod(size: Int): Int = ((this % size) + size) % size
    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun post(s: String) = runOnUiThread { status.text = s }

    /** Preview plus non-blocking advisory-cloud escalation from cross-exposure consistency. */
    private fun onFrameWithOffload(bmp: Bitmap, dets: List<Detection>, iso: Int, expUs: Int) {
        recordRotationFirstFramePose()
        showFrame(bmp, dets, iso, expUs)
        offloader?.takeIf { mc?.lastRoutingDecision?.shouldOffload == true }?.let { o ->
            val frameIdx = mc?.lastFrameIdx ?: 0
            displayIoExecutor.execute {
                val jpeg = ByteArrayOutputStream()
                    .also { bmp.compress(Bitmap.CompressFormat.JPEG, OffloadClient.JPEG_QUALITY, it) }
                    .toByteArray()
                o.offload(frameIdx, jpeg)
            }
        }
    }

    /** Validate the first real exposure-window midpoint, not the intentionally early trigger. */
    private fun recordRotationFirstFramePose() {
        val session = rotationSession ?: return
        if (!session.measurementStarted || !session.firstPoseRecorded.compareAndSet(false, true)) return
        val triggerController = session.trigger ?: return
        val runDir = session.logger?.dir ?: return
        val metas = raw.lastMeta
        if (metas.isEmpty()) return
        val first = metas.first()
        val last = metas.last()
        val midpointNs = (first.timestamp + last.timestamp + last.appliedExpUs * 1_000L) / 2L
        val pose = triggerController.poseAt(midpointNs)
        File(runDir, "rotation_first_frame.json").writeText(JSONObject().apply {
            put("first_sensor_timestamp_ns", first.timestamp)
            put("last_sensor_timestamp_ns", last.timestamp)
            put("exposure_window_midpoint_ns", midpointNs)
            put("pose_sample_timestamp_ns", pose?.sampleTimestampNs)
            put("pose_sample_offset_ns", pose?.sampleTimestampNs?.minus(midpointNs))
            put("center_error_deg", pose?.centerErrorDeg)
            put("gyro_y_rad_s", pose?.gyroY)
            put("direction", if ((pose?.gyroY ?: 0.0) > 0.0) "positive" else "negative")
            put("within_1deg", pose != null && pose.centerErrorDeg <= 1.0 && pose.gyroY > 0.0)
        }.toString(2))
        runCatching { triggerController.close() }
        session.trigger = null
    }

    /** Detection.xyxy is already in `bmp` pixel space, so the bitmap's own dims are what the
     *  overlay needs to scale against. */
    private fun showFrame(bmp: Bitmap, dets: List<Detection>, iso: Int, expUs: Int) {
        // Keep the 0.01 confidence tail in JSONL for offline analysis, but only draw boxes
        // that can actually contribute to the SoS selection score. This prevents All-COCO
        // low-confidence noise from looking like a valid on-glass detection.
        runOnUiThread {
            val drawn = dets.asSequence().filter { it.confidence >= displayConfThreshold }.map {
                val xy = it.xyxy
                OverlayView.DrawInfo(
                    Rect(xy[0].toInt(), xy[1].toInt(), xy[2].toInt(), xy[3].toInt()),
                    "${CocoLabels.name(it.classId)} %.2f".format(it.confidence))
            }.toList()
            val cellLabel = "ISO %d\nexp %.1f ms".format(iso, expUs / 1000.0)
            preview.setImageBitmap(bmp)
            overlay.setResults(drawn, bmp.width, bmp.height)
            cellText.text = cellLabel
        }
    }

    /** Runs [block] on the persistent GPU worker with a fresh logger and controller, so each
     *  press gets its own run directory while the camera and three GPU interpreters are reused. */
    private fun start(modeTag: String, block: () -> Unit) {
        if (runActive) { post("busy — stop first"); return }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) { post("no camera permission"); return }
        // All View reads happen here, on the UI thread, before the worker starts.
        val offUrl = (if (offloadCheck.isChecked) offloadUrl.text.toString().trim().ifBlank { null }
                      else null) ?: intent.getStringExtra("server_url")
        val offRegime = (offloadRegime.selectedItem?.toString() ?: "clear").ifBlank { "clear" }
        val selectConf = (confThreshSpinner.selectedItem?.toString()?.toFloatOrNull()) ?: 0.25f
        displayConfThreshold = selectConf
        runActive = true
        inferenceExecutor.execute {
            try {
                // A background transition invalidates camera ownership. This executes on the
                // same serial worker and therefore cannot race a GPU invocation.
                if (resourcesMustClose) {
                    closeResources()
                    resourcesMustClose = false
                }
                if (!opened) {
                    post("opening camera…")
                    raw = RawSensorCapturer(this); raw.open()
                    // Keep the paper's FP16 640px YOLOv8n and select the exact fixed batch for
                    // K=1/3/9. AUTO was validated on X3 Pro as OpenCL GPU delegate V2.
                    post("loading optimized COCO5 YOLO GPU batches…")
                    detector = TfliteYoloDetector(
                        this,
                        batchedAssets = listOf(
                            "yolov8n_640_coco5_fp16.tflite" to 1,
                            "yolov8n_640_b3_coco5_fp16.tflite" to 3
                        ),
                        numClasses = 5,
                        classIdMap = TfliteYoloDetector.COCO5_CLASS_IDS,
                        accelerator = TfliteYoloDetector.Accelerator.GPU,
                        allowFallback = false,
                        gpuBackend = TfliteYoloDetector.GpuBackend.AUTO,
                        onLoadStatus = ::post
                    )
                    post("GPU ready ${detector.backendSummary} — capturing first frame…")
                    opened = true
                }
                val runName = "run_${modeTag}_${System.currentTimeMillis()}"
                val logger = MeasurementLogger(this, runName)
                val mcLocal = MeasurementController(raw, detector, grid, sensors, logger, health, selectConf)
                mc = mcLocal
                // The currentFrame supplier reads lastFrameIdx, so staleness is measured
                // against the same frame index that frames.csv records.
                offloader = offUrl?.let {
                    post("offload -> $it ($offRegime)")
                    OffloadClient(it, logger.dir, offRegime) { mcLocal.lastFrameIdx }
                        .apply { warmConnection() }
                }
                try {
                    block()
                } finally {
                    logger.close()
                }
                // All 80 COCO classes are retained; the UI threshold only controls scoring/drawing.
                val above = mcLocal.detectionTotalAboveThresh
                val floor = mcLocal.detectionTotalAtFloor
                val frames = mcLocal.totalFramesLogged
                post("done — detections: $above above thresh (${floor} at floor), $frames frames")
                android.os.Handler(android.os.Looper.getMainLooper())
                    .postDelayed({ status.text = "" }, 5000)
            } catch (e: Throwable) {
                post("error: ${e.message}")
            } finally {
                runActive = false
            }
        }
    }

    override fun onResume() { super.onResume(); sensors.registerListeners() }
    override fun onPause() {
        super.onPause()
        cancelRotationSession()
        sensors.unregisterListeners()
        // Backgrounding revokes camera ownership, but Interpreter.close() must not race an
        // in-flight native run. Request stop and defer teardown until the worker has returned.
        mc?.stop()
        resourcesMustClose = true
        queueResourceClose()
    }

    @Suppress("DEPRECATION")
    override fun onBackPressed() {
        if (runActive) stopMeasurement() else super.onBackPressed()
    }

    override fun onDestroy() {
        cancelRotationSession()
        mc?.stop()
        resourcesMustClose = true
        queueResourceClose()
        inferenceExecutor.shutdown()
        displayIoExecutor.shutdown()
        health.close()
        sensors.release()
        super.onDestroy()
    }

    private fun queueResourceClose() {
        // FIFO ordering guarantees this runs after an in-flight detectBatch returns.
        runCatching {
            inferenceExecutor.execute {
                if (!resourcesMustClose) return@execute
                closeResources()
                resourcesMustClose = false
            }
        }
    }

    private fun closeResources() {
        if (!opened) return
        runCatching { raw.close() }
        runCatching { detector.close() }
        opened = false
    }
}
