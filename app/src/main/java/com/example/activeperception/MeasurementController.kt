package com.example.activeperception

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Paint
import android.os.Build
import android.os.SystemClock
import com.example.activeperception.acquire.StepResult
import com.example.activeperception.acquire.plan as acquirePlan
import com.example.activeperception.acquire.CandidateSource
import com.example.activeperception.acquire.Detection
import com.example.activeperception.acquire.Detector
import com.example.activeperception.acquire.Grid
import com.example.activeperception.acquire.CrossExposureRouter
import com.example.activeperception.acquire.RoutingDecision
import com.example.activeperception.acquire.RawFrame
import com.example.activeperception.acquire.RegimeClassifier
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future

/**
 * Tie-break used by Proposed when sum_conf is 0 on every candidate.
 *
 * ENTROPY        Shannon entropy of the luminance histogram. Brightness-biased: in dim
 *                scenes it picks the brightest cell regardless of blur.
 * LAPLACIAN_VAR  Variance of the 5-point Laplacian. Penalizes blur, but Lap² still scales
 *                with absolute brightness.
 * TENENGRAD_NORM Σ|∇I|² / Σ I² via Sobel. Brightness-invariant since both sums scale
 *                identically, and blur still costs gradient per unit brightness.
 * CRETE_ROFFET   Crete et al. 2007 perceptual blur metric.
 * SAFE_CELL      No image metric at all. With no detection anywhere the SNR is too low for
 *                a metric to discriminate, so pick by brightness target on the motion-safe
 *                shutter row instead.
 */
enum class FallbackMetric { ENTROPY, LAPLACIAN_VAR, TENENGRAD_NORM, CRETE_ROFFET, SAFE_CELL }

/**
 * Who chooses (iso, exposure) in the AE / AE_quant modes.
 * PHONE is the HAL's CONTROL_AE_MODE_ON — vendor-tuned and closed. CUSTOM_BRIGHTNESS is
 * [CustomAeBrightness], which is deterministic and reproducible. Quantization in AE_quant
 * is identical either way; only the upstream decision differs.
 */
enum class AeStrategy { PHONE, CUSTOM_BRIGHTNESS }

fun AeStrategy.tag(): String = when (this) {
    AeStrategy.PHONE -> "phone"
    AeStrategy.CUSTOM_BRIGHTNESS -> "custom"
}

/** Short run-dir suffix. */
fun FallbackMetric.tag(): String = when (this) {
    FallbackMetric.ENTROPY -> "ent"
    FallbackMetric.LAPLACIAN_VAR -> "lap"
    FallbackMetric.TENENGRAD_NORM -> "ten"
    FallbackMetric.CRETE_ROFFET -> "cre"
    FallbackMetric.SAFE_CELL -> "safe"
}
/** Status-tag display char. */
fun FallbackMetric.shortTag(): String = when (this) {
    FallbackMetric.ENTROPY -> "E"
    FallbackMetric.LAPLACIAN_VAR -> "L"
    FallbackMetric.TENENGRAD_NORM -> "T"
    FallbackMetric.CRETE_ROFFET -> "C"
    FallbackMetric.SAFE_CELL -> "S"
}
/** Value written into the candidates.csv `tie_break` column when this strategy was used. */
fun FallbackMetric.tieBreakName(): String = when (this) {
    FallbackMetric.ENTROPY -> "entropy"
    FallbackMetric.LAPLACIAN_VAR -> "laplacian"
    FallbackMetric.TENENGRAD_NORM -> "tenengrad"
    FallbackMetric.CRETE_ROFFET -> "crete_roffet"
    FallbackMetric.SAFE_CELL -> "safe_cell"
}

/**
 * Drives every measurement mode (Fixed | AE | AE_quant | Proposed | Verify | VerifyProbe |
 * Bench | IsoDiag) over the RAW capturer, the detector, and the acquire core. Runs on a
 * worker thread — capture and inference both block.
 *
 * Conventions shared by all modes:
 *  - the detector floors at conf 0.01, keeping a tail for the offload signal
 *  - selection re-thresholds at [selectConf]
 *  - the same on-device YOLOv8n runs everywhere
 *
 * Per-run file layout is in [MeasurementLogger].
 */
class MeasurementController(
    private val raw: RawSensorCapturer,
    private val detector: Detector<Bitmap>,
    private val grid: Grid,
    private val sensors: SensorDataManager,
    private val logger: MeasurementLogger,
    private val health: DeviceHealthMonitor,
    /** Operating confidence cutoff, above the detector's 0.01 decode floor. Applied to
     *  sum_conf in frames.csv and to Proposed's selection. Set from the UI, recorded in
     *  manifest as detector.select_conf_threshold. */
    private val selectConf: Float = 0.25f,
    private val colorPipeline: ColorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB
) {
    @Volatile var running = false; private set
    fun stop() { running = false }

    /** The canonical per-pass frame index: the `f` that names img/frame_XXXX.jpg and fills
     *  frames.csv `idx`. Published by [logFrame] just before each onFrame callback so
     *  external listeners (OffloadClient) can tag their work with a joinable id. */
    @Volatile var lastFrameIdx: Int = -1; private set
    @Volatile var lastRoutingDecision: RoutingDecision = RoutingDecision.KEEP_LOCAL; private set

    /** Per-pass totals for the post-run banner. Vehicle classes only — the detector already
     *  filters. The at-floor count includes the 0.01 tail, which is mostly noise. */
    @Volatile var detectionTotalAboveThresh: Int = 0; private set
    @Volatile var detectionTotalAtFloor: Int = 0; private set
    @Volatile var totalFramesLogged: Int = 0; private set

    /** Mean luma target for SAFE_CELL, as a fraction of white. 0.50 is the AE mid-gray
     *  reference: the linear midpoint of 8-bit display space, ≈ 18% gray after sRGB gamma. */
    private val SAFE_TARGET_RATIO = 0.50

    private val regime = RegimeClassifier()
    private var passStartMs: Long = 0L

    private fun now() = System.nanoTime()
    private fun ms(t0: Long) = (now() - t0) / 1e6

    // ---------- manifest + per-frame helpers shared by all modes ----------

    private fun headers() = listOf(
        "idx", "ts", "frame_number", "time_since_start", "lap", "heading_angle",
        "regime", "yaw_rate", "lux", "accel",
        "method", "chosen_cell", "gain", "exposure_us",
        "iso_req", "iso_applied", "exp_req", "exp_applied",
        "frame_duration_ns", "apply_delay_frames", "pending_commands",
        "black_r", "black_g1", "black_g2", "black_b", "row_stride", "pixel_stride",
        "wb_r", "wb_g", "wb_b",
        "ccm_00", "ccm_01", "ccm_02", "ccm_10", "ccm_11", "ccm_12",
        "ccm_20", "ccm_21", "ccm_22",
        "battery_temp_c", "thermal_status", "pss_kb",
        "K", "is_burst", "formation_ms", "infer_ms", "total_ms",
        "n_det", "sum_conf", "img_path",
        // Proposed only: which path picked the cell — conf | safe_cell | entropy |
        // laplacian | tenengrad | crete_roffet. Blank elsewhere.
        "tie_break",
        // AE_quant only: AE's continuous choice before quantization. The gain/exposure_us
        // columns above carry the quantized cell that was actually captured. Blank elsewhere.
        "ae_iso", "ae_exp_us"
    )

    /** [methodParams] carries the mode's own configuration so a run variant is identifiable
     *  from the manifest alone, without parsing the run directory name. Nested so the
     *  top-level schema stays stable as modes come and go. */
    private fun writeManifest(method: String, isGtReference: Boolean,
                              captureWidth: Int, captureHeight: Int,
                              methodParams: JSONObject? = null) {
        val m = JSONObject()
        m.put("method", method)
        m.put("is_gt_reference", isGtReference)
        m.put("pass_start_ts", System.currentTimeMillis())
        m.put("device_model", Build.MODEL)
        m.put("os_sdk", Build.VERSION.SDK_INT)
        if (methodParams != null) m.put("method_params", methodParams)
        m.put("grid", JSONObject().apply {
            put("gains", JSONArray(grid.gains.toTypedArray()))
            put("exposures_us", JSONArray(grid.exposuresUs.toTypedArray()))
            put("n_gain", grid.nGain); put("n_shutter", grid.nShutter)
            // Uniform digital multiplier folded into gainRatio; see Grid.
            put("digital_boost", grid.digitalBoost)
        })
        m.put("capture_resolution", JSONObject().apply {
            put("width", captureWidth); put("height", captureHeight)
            put("downsampled_in_decode", true)
            // The RAW size above is NOT the space the logged boxes live in — decode()
            // block-skips 2× and demosaic2x2 halves again. Every xyxy in dets.jsonl and
            // candidate_dets.jsonl is in the detector_input space recorded here.
            put("detector_input_width", raw.bitmapWidth)
            put("detector_input_height", raw.bitmapHeight)
        })
        m.put("cfa_pattern", raw.cfaPattern)
        m.put("white_level", raw.maxDn)
        m.put("sync_max_latency", raw.syncMaxLatency)
        m.put("timestamp_source", raw.timestampSource)
        m.put("sensor_orientation", raw.sensorOrientation)
        m.put("image_formation", JSONObject().apply {
            put("pipeline", colorPipeline.tag)
            put("orientation_applied_clockwise_degrees", raw.sensorOrientation)
            put("white_balance", if (colorPipeline == ColorPipeline.RAYNEO_AWB_CCM)
                "per-frame CaptureResult.COLOR_CORRECTION_GAINS" else "not applied")
            put("color_transform", if (colorPipeline == ColorPipeline.RAYNEO_AWB_CCM)
                "per-frame CaptureResult.COLOR_CORRECTION_TRANSFORM" else "not applied")
            put("order", if (colorPipeline == ColorPipeline.RAYNEO_AWB_CCM)
                "black subtract -> burst sum -> demosaic -> digital gain -> AWB -> CCM -> sRGB OETF -> rotate"
                else "black subtract -> burst sum -> demosaic -> digital gain -> sRGB OETF -> rotate")
        })
        m.put("rayneo_profile", JSONObject().apply {
            put("camera_id", "0")
            put("measured_frame_period_ns", 33_329_000L)
            put("measured_setting_delay_frames", "9-10")
            put("setting_guard_frames", 0)
            put("setting_acceptance", "first timestamp-paired RAW with matching exposure/ISO metadata")
            put("metadata_verified_actuation", true)
            put("virtual_window", if (colorPipeline == ColorPipeline.ORIGINAL_GAIN_SRGB)
                "first_n_original_sos" else "centered_rayneo_legacy")
        })
        // Mirrors the TfliteYoloDetector defaults the Activity constructs it with. Boxes in
        // dets.jsonl are post-NMS, post-class-filter, post-conf_floor.
        m.put("detector", JSONObject().apply {
            put("model", "yolov8n_640_coco5_fp16")
            put("img_size", 640)
            put("conf_floor", 0.01)
            put("iou_thresh", 0.45)
            put("max_det_per_frame", 100)
            put("class_filter", "coco5_head")
            put("allowed_classes", JSONArray(listOf(41, 40, 46, 5, 60)))
            put("class_names", JSONArray(listOf("cup", "wine glass", "banana", "bus", "dining table")))
            put("num_classes", 5)
            put("select_conf_threshold", selectConf)
            put("backend_by_batch",
                (detector as? com.example.activeperception.acquire.TfliteYoloDetector)
                    ?.backendSummary ?: "unknown")
        })
        logger.manifest(m)
    }

    private fun sumConf(dets: List<Detection>, sel: Float = selectConf) =
        dets.sumOf { if (it.confidence >= sel) it.confidence.toDouble() else 0.0 }

    private fun detJson(d: Detection): JSONObject {
        val xy = JSONArray()
        for (v in d.xyxy) xy.put(v.toDouble())
        return JSONObject().apply {
            put("xyxy", xy); put("conf", d.confidence.toDouble()); put("cls", d.classId)
        }
    }

    /** Full detection list including the sub-threshold tail. `method` keeps rows separable
     *  when a mode writes more than one per frame. */
    private fun writeDets(frame: Int, method: String, cell: Int, dets: List<Detection>) {
        val arr = JSONArray()
        for (d in dets) arr.put(detJson(d))
        logger.jsonl("dets", JSONObject().apply {
            put("frame", frame); put("method", method); put("cell", cell); put("dets", arr)
        })
    }

    /** Snap a continuous (iso, expUs) onto the discrete action set, each axis independently,
     *  by smallest |ln(target / grid)|. Log distance because the axes are geometric. */
    private fun quantizeToGrid(iso: Int, expUs: Int): Pair<Int, Int> {
        val isoSafe = iso.coerceAtLeast(1).toDouble()
        val expSafe = expUs.coerceAtLeast(1).toDouble()
        val gi = grid.gains.indices.minByOrNull {
            kotlin.math.abs(kotlin.math.ln(isoSafe / grid.gains[it]))
        } ?: 0
        val sj = grid.exposuresUs.indices.minByOrNull {
            kotlin.math.abs(kotlin.math.ln(expSafe / grid.exposuresUs[it]))
        } ?: 0
        return Pair(gi, sj)
    }

    /** One imu.csv row per sensor sample, tagged by `source`, carrying only that sensor's
     *  channels and its own ns timestamp. Merging channels into one row would attach a stale
     *  timestamp to whichever sensor did not fire, which offline orientation fusion can't undo. */
    private fun wireImuLog() {
        logger.csv("imu", listOf("ts", "source", "ax", "ay", "az", "gx", "gy", "gz", "lux"))
        sensors.onAccelSample = { ts, ax, ay, az ->
            logger.row("imu", listOf(ts, "accel", ax, ay, az, "", "", "", ""))
        }
        sensors.onGyroSample = { ts, gx, gy, gz ->
            logger.row("imu", listOf(ts, "gyro", "", "", "", gx, gy, gz, ""))
        }
        sensors.onLuxSample = { ts, lux ->
            logger.row("imu", listOf(ts, "light", "", "", "", "", "", "", lux))
        }
    }

    private fun unwireImuLog() {
        sensors.onAccelSample = null
        sensors.onGyroSample = null
        sensors.onLuxSample = null
    }

    /**
     * Shared per-frame row writer.
     *
     * Two different ISO conventions land in the same row on purpose: `gain` is EFFECTIVE ISO
     * (physical × digitalBoost, matching the saved bitmap), while `iso_req` / `iso_applied`
     * stay physical so the camera metadata is faithful. digital_boost is in the manifest, so
     * physical = effective / boost.
     *
     * Note that `iso_req` / `exp_req` describe the capture that produced the pixels, which in
     * Proposed is the base capture — not the virtual cell named by `gain` / `exposure_us`.
     */
    private fun logFrame(
        idx: Int, method: String, chosenCell: Int, gainVal: Int, expUs: Int,
        k: Int, isBurst: Boolean, formationMs: Double, inferMs: Double, totalMs: Double,
        dets: List<Detection>, imgPath: String, tieBreak: String = "",
        aeIso: Int = -1, aeExpUs: Int = -1
    ) {
        // Published before the mode's onFrame callback fires, so listeners see the matching id.
        lastFrameIdx = idx
        detectionTotalAtFloor += dets.size
        detectionTotalAboveThresh += dets.count { it.confidence >= selectConf }
        totalFramesLogged += 1
        val m = raw.lastMeta.firstOrNull()
        val aeIsoCol = if (aeIso < 0) "" else aeIso.toString()
        val aeExpCol = if (aeExpUs < 0) "" else aeExpUs.toString()
        val healthNow = health.sample()
        logger.row("frames", listOf(
            idx, m?.timestamp ?: -1L, m?.frameNumber ?: -1L,
            "%.3f".format((System.currentTimeMillis() - passStartMs) / 1000.0),
            sensors.getCurrentLap(), "%.3f".format(sensors.currentHeadingAngle),
            regime.update(sensors.currentYawRate),
            sensors.currentYawRate, sensors.currentLux, sensors.currentAccel,
            method, chosenCell, effIso(gainVal), expUs,
            m?.requestedIso ?: -1, m?.appliedIso ?: -1,
            m?.requestedExpUs ?: -1L, m?.appliedExpUs ?: -1L,
            m?.frameDurationNs ?: -1L, m?.applyDelayFrames ?: raw.lastApplyDelayFrames ?: -1L,
            raw.pendingCommandCount,
            m?.blackLevels?.getOrNull(0) ?: -1, m?.blackLevels?.getOrNull(1) ?: -1,
            m?.blackLevels?.getOrNull(2) ?: -1, m?.blackLevels?.getOrNull(3) ?: -1,
            m?.rowStrideBytes ?: -1, m?.pixelStrideBytes ?: -1,
            m?.whiteBalance?.getOrNull(0) ?: -1f,
            m?.whiteBalance?.getOrNull(1) ?: -1f,
            m?.whiteBalance?.getOrNull(2) ?: -1f,
            *(m?.cameraToSrgb ?: DoubleArray(9) { -1.0 }).toTypedArray(),
            healthNow.batteryTemperatureC, healthNow.thermalStatus, healthNow.totalPssKb,
            k, if (isBurst) 1 else 0,
            "%.1f".format(formationMs), "%.1f".format(inferMs), "%.1f".format(totalMs),
            dets.size, "%.3f".format(sumConf(dets)), imgPath, tieBreak,
            aeIsoCol, aeExpCol
        ))
        writeDets(idx, method, chosenCell, dets)
    }

    // ---------- Fixed ----------

    /** Every frame captured at one physical (gain, exposure) cell, AE off. */
    fun runFixed(gainIdx: Int, shutterIdx: Int, maxFrames: Int, isGtReference: Boolean,
                 onStatus: (String) -> Unit,
                 onFrame: (Bitmap, List<Detection>, Int /*iso*/, Int /*expUs*/) -> Unit = { _, _, _, _ -> }) {
        val gainVal = grid.gains[gainIdx]; val expUs = grid.exposuresUs[shutterIdx]
        val cell = grid.cell(gainIdx, shutterIdx)
        startPass("fixed_g${gainVal}_e${expUs}", isGtReference, methodParams = JSONObject()
            .put("gain_idx", gainIdx).put("shutter_idx", shutterIdx).put("cell", cell))
        var f = 0
        while (running && f < maxFrames) {
            val t0 = now()
            val frames = raw.capture(expUs, gainVal, 1)
            if (frames.isEmpty()) break
            val tf = now()
            val bmp = bitmapFromRaw(frames[0])
            val formMs = ms(tf)
            val ti = now(); val dets = detector.detectBatch(listOf(bmp))[0]; val infMs = ms(ti)
            val totalMs = ms(t0)
            val path = logger.saveJpegAsync("${frameName(f)}_${isoExpTag(effIso(gainVal), expUs)}", bmp)
            logFrame(f, "fixed", cell, gainVal, expUs, 1, false, formMs, infMs, totalMs, dets, path)
            onFrame(bmp, dets, effIso(gainVal), expUs)
            onStatus("Fixed f=$f g=${effIso(gainVal)} exp=${expUs}us ndet=${dets.size} ${"%.0f".format(totalMs)}ms")
            f++
        }
        endPass()
    }

    // ---------- AE ----------

    /** Every frame captured under [aeStrategy], logging whichever (iso, exp) it chose. */
    fun runAe(maxFrames: Int, isGtReference: Boolean,
              aeStrategy: AeStrategy,
              onStatus: (String) -> Unit,
              onFrame: (Bitmap, List<Detection>, Int /*iso*/, Int /*expUs*/) -> Unit = { _, _, _, _ -> }) {
        val customAe = if (aeStrategy == AeStrategy.CUSTOM_BRIGHTNESS) CustomAeBrightness() else null
        startPass("ae_${aeStrategy.tag()}", isGtReference, methodParams = JSONObject()
            .put("ae_strategy", aeStrategy.tag())
            .apply { if (customAe != null) put("custom_ae", customAe.toJson()) })
        // Custom AE seed; replaced every frame by the brightness feedback below.
        var nextIso = grid.baseGain
        var nextExpUs = grid.fastestExposureUs
        var f = 0
        while (running && f < maxFrames) {
            val t0 = now()
            val frames = when (aeStrategy) {
                AeStrategy.PHONE -> raw.captureAe(1)
                AeStrategy.CUSTOM_BRIGHTNESS -> raw.capture(nextExpUs, nextIso, 1)
            }
            if (frames.isEmpty()) break
            val tf = now()
            val bmp = bitmapFromRaw(frames[0])
            val formMs = ms(tf)
            val ti = now(); val dets = detector.detectBatch(listOf(bmp))[0]; val infMs = ms(ti)
            val totalMs = ms(t0)
            val gainVal: Int; val expUs: Int
            when (aeStrategy) {
                AeStrategy.PHONE -> {
                    val applied = raw.lastMeta.firstOrNull()
                    gainVal = applied?.appliedIso ?: -1
                    expUs = (applied?.appliedExpUs ?: -1L).toInt()
                }
                AeStrategy.CUSTOM_BRIGHTNESS -> {
                    gainVal = nextIso
                    expUs = nextExpUs
                }
            }
            val path = logger.saveJpegAsync("${frameName(f)}_${isoExpTag(effIso(gainVal), expUs)}", bmp)
            logFrame(f, "ae", -1, gainVal, expUs, 1, false, formMs, infMs, totalMs, dets, path)
            onFrame(bmp, dets, effIso(gainVal), expUs)
            onStatus("AE[${aeStrategy.tag()}] f=$f iso=${effIso(gainVal)} exp=${expUs}us ndet=${dets.size} ${"%.0f".format(totalMs)}ms")
            if (customAe != null) {
                val ratio = meanRawRatio(frames[0])
                val (newIso, newExp) = customAe.next(gainVal, expUs, ratio)
                nextIso = newIso; nextExpUs = newExp
            }
            f++
        }
        endPass()
    }

    // ---------- AE_quant (AE restricted to the grid's action set) ----------

    /** AE picks (iso, exp), that choice is quantized to the nearest grid cell, and only the
     *  quantized cell is physically captured and detected on. AE's unquantized choice is kept
     *  as metadata in frames.csv (`ae_iso`, `ae_exp_us`).
     *
     *  With PHONE strategy this alternates AE-on and manual captures, which disturbs AE's
     *  convergence further than [runAe] already does. */
    fun runAeQuant(maxFrames: Int, isGtReference: Boolean,
                   aeStrategy: AeStrategy,
                   onStatus: (String) -> Unit,
                   onFrame: (Bitmap, List<Detection>, Int /*iso*/, Int /*expUs*/) -> Unit = { _, _, _, _ -> }) {
        val customAe = if (aeStrategy == AeStrategy.CUSTOM_BRIGHTNESS) CustomAeBrightness() else null
        startPass("ae_paired_${aeStrategy.tag()}", isGtReference, methodParams = JSONObject()
            .put("ae_strategy", aeStrategy.tag())
            .apply { if (customAe != null) put("custom_ae", customAe.toJson()) })
        var nextIso = grid.baseGain
        var nextExpUs = grid.fastestExposureUs
        var f = 0
        while (running && f < maxFrames) {
            val t0 = now()
            // Phone AE only reveals its choice through a real capture; Custom AE computes it.
            val aeIso: Int; val aeExpUs: Int
            val aeRawForFeedback: RawFrame?
            when (aeStrategy) {
                AeStrategy.PHONE -> {
                    val aeFrames = raw.captureAe(1)
                    if (aeFrames.isEmpty()) break
                    val aeMeta = raw.lastMeta.firstOrNull()
                    aeIso = aeMeta?.appliedIso ?: -1
                    aeExpUs = (aeMeta?.appliedExpUs ?: -1L).toInt()
                    aeRawForFeedback = aeFrames[0]
                }
                AeStrategy.CUSTOM_BRIGHTNESS -> {
                    aeIso = nextIso; aeExpUs = nextExpUs
                    aeRawForFeedback = null   // feedback comes from the quantized frame instead
                }
            }

            if (!running) break

            val (qGi, qSj) = quantizeToGrid(aeIso, aeExpUs)
            val qGain = grid.gains[qGi]; val qExp = grid.exposuresUs[qSj]; val qCell = grid.cell(qGi, qSj)
            val qFrames = raw.capture(qExp, qGain, 1)
            if (qFrames.isEmpty()) break
            val tfQ = now()
            val bmpQ = bitmapFromRaw(qFrames[0])
            val formQMs = ms(tfQ)
            val tiQ = now(); val detsQ = detector.detectBatch(listOf(bmpQ))[0]; val infQMs = ms(tiQ)
            val totalMs = ms(t0)
            val pathQ = logger.saveJpegAsync("frame_%04d_aequant_%s".format(f, isoExpTag(effIso(qGain), qExp)), bmpQ)
            logFrame(f, "ae_quant", qCell, qGain, qExp, 1, false, formQMs, infQMs, totalMs,
                detsQ, pathQ, aeIso = aeIso, aeExpUs = aeExpUs)

            onFrame(bmpQ, detsQ, effIso(qGain), qExp)
            onStatus("AE_quant[${aeStrategy.tag()}] f=$f AE→(iso=${effIso(aeIso)}, exp=${aeExpUs}us) → cell=$qCell ${"%.0f".format(totalMs)}ms")
            // Prefer AE's own capture for brightness feedback — it is closer to the exposure
            // AE actually asked for than the quantized one is.
            if (customAe != null) {
                val src = aeRawForFeedback ?: qFrames[0]
                val ratio = meanRawRatio(src)
                val (newIso, newExp) = customAe.next(aeIso, aeExpUs, ratio)
                nextIso = newIso; nextExpUs = newExp
            }
            f++
        }
        endPass()
    }

    // ---------- Proposed (acquire-and-select) ----------

    /** Keeps the rendered bitmaps and the render time visible without changing acquire/. */
    private class RecordingSource(private val inner: CandidateSource<Bitmap>) : CandidateSource<Bitmap> {
        var lastImages: List<Bitmap> = emptyList(); private set
        var lastRenderMs: Double = 0.0; private set
        override fun render(cells: IntArray): List<Bitmap> {
            val t0 = System.nanoTime()
            val imgs = inner.render(cells)
            lastRenderMs = (System.nanoTime() - t0) / 1e6
            lastImages = imgs
            return imgs
        }
    }
    private class TimingDetector(private val inner: Detector<Bitmap>) : Detector<Bitmap> {
        var lastInferMs: Double = 0.0; private set
        override fun detectBatch(images: List<Bitmap>): List<List<Detection>> {
            val t0 = System.nanoTime()
            val out = inner.detectBatch(images)
            lastInferMs = (System.nanoTime() - t0) / 1e6
            return out
        }
    }

    /** Lazy so it is not spun up unless a fallback metric is actually used. */
    private val fallbackPool by lazy { java.util.concurrent.Executors.newFixedThreadPool(4) }

    /** Per-candidate scores in input order. SAFE_CELL scores nothing — the caller picks by
     *  brightness target instead — so it returns zeros. */
    private fun fallbackScores(images: List<Bitmap>, metric: FallbackMetric): DoubleArray {
        if (metric == FallbackMetric.SAFE_CELL) return DoubleArray(images.size)
        val fn: (Bitmap) -> Double = when (metric) {
            FallbackMetric.ENTROPY -> ::pixelEntropy
            FallbackMetric.LAPLACIAN_VAR -> ::laplacianVariance
            FallbackMetric.TENENGRAD_NORM -> ::tenengradNorm
            FallbackMetric.CRETE_ROFFET -> ::creteRoffet
            FallbackMetric.SAFE_CELL -> error("unreachable")
        }
        if (images.size <= 1) return DoubleArray(images.size) { fn(images[it]) }
        val futures = images.map { bmp -> fallbackPool.submit<Double> { fn(bmp) } }
        return DoubleArray(images.size) { futures[it].get() }
    }

    /** Shannon entropy of the luminance histogram, 0..8 bits. See [FallbackMetric]. */
    private fun pixelEntropy(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        val hist = IntArray(256)
        for (p in px) {
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            val lum = ((r * 299 + g * 587 + b * 114) / 1000).coerceIn(0, 255)
            hist[lum]++
        }
        val total = px.size.toDouble()
        val ln2 = Math.log(2.0)
        var ent = 0.0
        for (c in hist) if (c > 0) { val q = c / total; ent -= q * (Math.log(q) / ln2) }
        return ent
    }

    /** Variance of the 5-point Laplacian of luminance (Pech-Pacheco et al.; Krotkov
     *  autofocus). See [FallbackMetric]. Single pass: var = E[X²] - E[X]². */
    private fun laplacianVariance(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        // Luminance precomputed so the stencil doesn't unpack RGB 4× per inner pixel.
        val lum = IntArray(w * h)
        for (i in px.indices) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            lum[i] = (r * 299 + g * 587 + b * 114) / 1000
        }
        var sum = 0.0; var sumSq = 0.0; var n = 0
        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val c = lum[row + x]
                val l = lum[row + x - 1]; val rr = lum[row + x + 1]
                val u = lum[row - w + x]; val d = lum[row + w + x]
                val v = (l + rr + u + d - 4 * c).toDouble()
                sum += v; sumSq += v * v; n++
            }
        }
        if (n == 0) return 0.0
        val mean = sum / n
        return sumSq / n - mean * mean
    }

    /** Normalized Tenengrad: Σ|∇I|² / Σ I² via Sobel 3×3. See [FallbackMetric]. */
    private fun tenengradNorm(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        val lum = IntArray(w * h)
        for (i in px.indices) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            lum[i] = (r * 299 + g * 587 + b * 114) / 1000
        }
        var sumG2 = 0.0; var sumI2 = 0.0
        for (y in 1 until h - 1) {
            val row = y * w
            val rowU = row - w; val rowD = row + w
            for (x in 1 until w - 1) {
                // Sobel 3×3, Gx = [-1 0 1; -2 0 2; -1 0 1], Gy transposed.
                val gx = -lum[rowU + x - 1] + lum[rowU + x + 1] +
                         -2 * lum[row  + x - 1] + 2 * lum[row  + x + 1] +
                         -lum[rowD + x - 1] + lum[rowD + x + 1]
                val gy = -lum[rowU + x - 1] - 2 * lum[rowU + x] - lum[rowU + x + 1] +
                          lum[rowD + x - 1] + 2 * lum[rowD + x] + lum[rowD + x + 1]
                sumG2 += (gx * gx + gy * gy).toDouble()
                val v = lum[row + x].toDouble()
                sumI2 += v * v
            }
        }
        return if (sumI2 == 0.0) 0.0 else sumG2 / sumI2
    }

    /** Crete-Roffet 2007 no-reference blur metric, returned as SHARPNESS so that, like every
     *  other metric here, higher is better. Re-blurs with a 9-tap separable box filter and
     *  measures the gradient lost: a sharp image loses a lot, an already-blurred one little.
     *  Sliding-sum box blur, so each pass is O(N) regardless of radius. */
    private fun creteRoffet(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        if (w < 3 || h < 3) return 0.0
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        val I = FloatArray(w * h)
        for (i in px.indices) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            I[i] = ((r * 299 + g * 587 + b * 114) / 1000).toFloat()
        }
        val radius = 4   // 9-tap box (Crete-Roffet paper)
        val tmp = FloatArray(w * h)
        val B = FloatArray(w * h)
        // Horizontal box blur (sliding sum, edge-clamped count)
        for (y in 0 until h) {
            val rowOff = y * w
            var sum = 0f
            val initEnd = minOf(radius, w - 1)
            for (k in 0..initEnd) sum += I[rowOff + k]
            var count = initEnd + 1
            tmp[rowOff] = sum / count
            for (x in 1 until w) {
                val addIdx = x + radius; val rmvIdx = x - radius - 1
                if (addIdx < w) { sum += I[rowOff + addIdx]; count++ }
                if (rmvIdx >= 0) { sum -= I[rowOff + rmvIdx]; count-- }
                tmp[rowOff + x] = sum / count
            }
        }
        // Vertical box blur
        for (x in 0 until w) {
            var sum = 0f
            val initEnd = minOf(radius, h - 1)
            for (k in 0..initEnd) sum += tmp[k * w + x]
            var count = initEnd + 1
            B[x] = sum / count
            for (y in 1 until h) {
                val addIdx = y + radius; val rmvIdx = y - radius - 1
                if (addIdx < h) { sum += tmp[addIdx * w + x]; count++ }
                if (rmvIdx >= 0) { sum -= tmp[rmvIdx * w + x]; count-- }
                B[y * w + x] = sum / count
            }
        }
        // Horizontal and vertical diff comparison
        var sumDh = 0.0; var sumVh = 0.0
        for (y in 0 until h) {
            val rowOff = y * w
            for (x in 1 until w) {
                val dh = Math.abs(I[rowOff + x] - I[rowOff + x - 1])
                val dhB = Math.abs(B[rowOff + x] - B[rowOff + x - 1])
                sumDh += dh; sumVh += maxOf(0f, dh - dhB)
            }
        }
        var sumDv = 0.0; var sumVv = 0.0
        for (y in 1 until h) {
            val rowOff = y * w; val rowOffU = rowOff - w
            for (x in 0 until w) {
                val dv = Math.abs(I[rowOff + x] - I[rowOffU + x])
                val dvB = Math.abs(B[rowOff + x] - B[rowOffU + x])
                sumDv += dv; sumVv += maxOf(0f, dv - dvB)
            }
        }
        // ratio = sumV/sumD ∈ [0,1] is the fraction of gradient lost to re-blurring, so it
        // is ≈1 for a sharp image and ≈0 for a blurred one. The paper's blur_F is
        // max(1-ratioH, 1-ratioV), hence sharpness = 1 - blur_F = min(ratioH, ratioV).
        val ratioH = if (sumDh > 0) sumVh / sumDh else 0.0
        val ratioV = if (sumDv > 0) sumVv / sumDv else 0.0
        return minOf(ratioH, ratioV)
    }

    /**
     * The deployed acquire-select loop. Same as `AcquireSelectController` except for the
     * all-zero case: instead of holding the anchor when no candidate has any detection, it
     * picks by [metric] and anchors there, turning a dead step into exploration. That matters
     * on cold start and through brief occlusions.
     *
     * `acquire/AcquireSelectController` is deliberately left untouched as the reference.
     */
    private inner class EntropyFallbackController(
        private val source: CandidateSource<Bitmap>,
        private val detector: Detector<Bitmap>,
        private val period: Int,
        initAnchor: Int,
        private val metric: FallbackMetric = FallbackMetric.ENTROPY,
        private val selectConf: Float = this@MeasurementController.selectConf
    ) {
        var anchor: Int = initAnchor; private set
        var t: Int = 0; private set
        /** Empty when the conf path picked the cell. */
        var lastFallbackScores: DoubleArray = DoubleArray(0); private set
        var lastUsedFallback: Boolean = false; private set
        /** Every candidate's detections, indexed parallel to StepResult.cells, so the caller
         *  can log all of them and not just the chosen one. */
        var lastAllDets: List<List<Detection>> = emptyList(); private set

        fun step(): StepResult {
            val cells = acquirePlan(t, anchor, grid, period)
            val images = source.render(cells)
            val dets = detector.detectBatch(images)
            lastAllDets = dets
            val confScores = DoubleArray(dets.size) { i ->
                dets[i].sumOf { if (it.confidence >= selectConf) it.confidence.toDouble() else 0.0 }
            }
            var chosen = 0
            for (i in confScores.indices) if (confScores[i] > confScores[chosen]) chosen = i

            lastFallbackScores = DoubleArray(0); lastUsedFallback = false
            if (confScores[chosen] == 0.0) {
                lastUsedFallback = true
                if (metric == FallbackMetric.SAFE_CELL) {
                    // Among shutter-row-0 cells (no motion-blur risk), take the one whose mean
                    // luma is nearest the target in log distance. That adapts: dim scenes land
                    // on high gain, bright ones on low gain instead of saturating.
                    val target = SAFE_TARGET_RATIO
                    var pick = -1; var bestDist = Double.MAX_VALUE
                    for (i in cells.indices) {
                        if (grid.indices(cells[i]).second != 0) continue
                        val mean = meanLumaRatio(images[i])
                        val d = kotlin.math.abs(kotlin.math.ln((mean + 1e-6) / target))
                        if (d < bestDist) { bestDist = d; pick = i }
                    }
                    if (pick < 0) {
                        // Off-probe step anchored off row 0, so no row-0 cell is a candidate.
                        // Fall back to the highest gain available; the next probe step brings
                        // row 0 back into range.
                        var bestGi = -1
                        for (i in cells.indices) {
                            val gi = grid.indices(cells[i]).first
                            if (gi > bestGi) { bestGi = gi; pick = i }
                        }
                        if (pick < 0) pick = 0
                    }
                    chosen = pick
                } else {
                    val scores = fallbackScores(images, metric)
                    lastFallbackScores = scores
                    chosen = 0
                    for (i in scores.indices) if (scores[i] > scores[chosen]) chosen = i
                }
            }

            anchor = cells[chosen]
            t += 1
            return StepResult(cells[chosen], dets[chosen], confScores, cells, chosen)
        }
    }

    /** Live acquire-select controller. Writes frames.csv + dets.jsonl + candidates.csv + img/.
     *  [fallback] selects which tie-break metric is used when sum_conf=0 across all cells.
     *  Full-grid steps use one base burst and virtual formation. Gain-column steps use the
     *  metadata-confirmed physical shutter row selected by the previous step. */
    fun runProposed(period: Int, maxFrames: Int, isGtReference: Boolean,
                    fallback: FallbackMetric,
                    /** Saves a bitmap for every candidate, not just the chosen one — useful for
                     *  visual inspection, but roughly +50% wall-clock on probe-heavy runs.
                     *  candidate_dets.jsonl records all per-cell boxes either way. */
                    saveAllCandidates: Boolean = false,
                    onStatus: (String) -> Unit,
                    onFrame: (Bitmap, List<Detection>, Int /*iso*/, Int /*expUs*/) -> Unit = { _, _, _, _ -> }) {
        val initAnchor = grid.cell(grid.nGain - 1, 0)
        startPass("proposed_p${period}_${fallback.tag()}", isGtReference,
            methodParams = JSONObject()
                .put("period", period)
                .put("fallback_metric", fallback.tieBreakName())
                .put("init_anchor_cell", initAnchor)
                .put("save_all_candidates", saveAllCandidates))
        // fallback_score is blank unless the fallback fired; tie_break names the path that
        // picked the cell.
        logger.csv("candidates", listOf("frame", "cell", "gain", "exposure_us",
            "sum_conf", "fallback_score", "chosen", "tie_break"))
        logger.csv("router", listOf("frame", "score", "model_limited_clusters",
            "recovered_locally_clusters", "offload"))
        val source = RecordingSource(ParallelRawCandidateSource(grid, raw))
        val timing = TimingDetector(detector)
        // Anchor starts at (max gain, fastest shutter) so the very first preview is visible
        // whatever the lighting.
        val ctrl = EntropyFallbackController(source, timing, period, initAnchor, fallback)
        val router = CrossExposureRouter(selectConf)
        val fbName = fallback.tieBreakName()
        val fbShort = fallback.shortTag()
        var f = 0
        while (running && f < maxFrames) {
            val t0 = now(); val r = ctrl.step(); val totalMs = ms(t0)
            val isBurst = r.cells.size > grid.nGain
            val tieBreak = if (ctrl.lastUsedFallback) fbName else "conf"
            val fbScores = ctrl.lastFallbackScores
            val allDets = ctrl.lastAllDets
            lastRoutingDecision = router.decide(allDets)
            logger.row("router", listOf(f, "%.5f".format(lastRoutingDecision.score),
                lastRoutingDecision.modelLimitedClusters,
                lastRoutingDecision.recoveredLocallyClusters,
                if (lastRoutingDecision.shouldOffload) 1 else 0))
            var chosenPath = ""
            for (i in r.cells.indices) {
                val c = r.cells[i]; val (gi, sj) = grid.indices(c)
                val gainEff = effIso(grid.gains[gi]); val expU = grid.exposuresUs[sj]
                val scoreCell = if (i < fbScores.size) "%.3f".format(fbScores[i]) else ""
                logger.row("candidates", listOf(
                    f, c, gainEff, expU,
                    "%.4f".format(r.scores[i]), scoreCell,
                    if (r.cells[i] == r.cell) 1 else 0, tieBreak))
                val candBmp = source.lastImages.getOrNull(i) ?: continue
                val isChosen = (i == r.chosen)
                if (saveAllCandidates || isChosen) {
                    val candPath = logger.saveJpegAsync(
                        "${frameName(f)}_cell${c}_${isoExpTag(gainEff, expU)}", candBmp)
                    if (isChosen) chosenPath = candPath
                }
                // Joinable with frames.csv on `frame` and candidates.csv on (frame, cell).
                val detsArr = JSONArray()
                val cellDets = if (i < allDets.size) allDets[i] else emptyList()
                for (d in cellDets) detsArr.put(detJson(d))
                logger.jsonl("candidate_dets", JSONObject().apply {
                    put("frame", f); put("cell", c); put("chosen", isChosen)
                    put("gain", gainEff); put("exposure_us", expU)
                    put("sum_conf", r.scores[i])
                    put("tie_break", if (isChosen) tieBreak else "")
                    put("dets", detsArr)
                })
            }
            val (gi, sj) = grid.indices(r.cell)
            val gainVal = grid.gains[gi]; val expUs = grid.exposuresUs[sj]
            val bmp = source.lastImages.getOrNull(r.chosen)
            logFrame(f, "proposed", r.cell, gainVal, expUs,
                r.cells.size, isBurst, source.lastRenderMs, timing.lastInferMs, totalMs,
                r.detections, chosenPath, tieBreak)
            if (bmp != null) onFrame(bmp, r.detections, effIso(gainVal), expUs)
            val tag = if (ctrl.lastUsedFallback) " [$fbShort]" else ""
            onStatus("Proposed f=$f cell=${r.cell}$tag ndet=${r.detections.size} ${"%.0f".format(totalMs)}ms (form=${"%.0f".format(source.lastRenderMs)}, inf=${"%.0f".format(timing.lastInferMs)})")
            f++
        }
        endPass()
    }

    // ---------- Verify: physical half of the probing-realism pair ----------

    /** Physically visits every cell twice on a static scene, saving the RAW, a formed JPEG,
     *  and the detections for each. Pair with [runVerifyProbe] on the same scene to compare
     *  physically-visited cells against virtually-formed ones.
     *
     *  Both passes exist so the physical-vs-physical difference gives a null baseline for
     *  the physical-vs-virtual comparison. The detector runs here so the comparison can be
     *  made from the CSVs alone, without a round trip through offline Python. */
    fun runVerify(isGtReference: Boolean, onStatus: (String) -> Unit,
                  onFrame: (Bitmap, List<Detection>, Int /*iso*/, Int /*expUs*/) -> Unit = { _, _, _, _ -> }) {
        startPass("verify", isGtReference, withFrames = false)
        logger.csv("verify", listOf(
            "ts", "scene", "pass", "cell", "gain", "iso_applied", "exp_applied",
            "black", "white", "raw_path", "img_path", "n_det", "sum_conf"))
        val base = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        if (base.isNotEmpty()) {
            val p = logger.saveRaw("verify_simbase_${isoExpTag(effIso(grid.baseGain), grid.fastestExposureUs)}", base[0].bayer)
            val m = raw.lastMeta.firstOrNull()
            logger.row("verify", listOf(System.currentTimeMillis(), "static", "sim_base", -1,
                effIso(grid.baseGain), m?.appliedIso ?: -1, m?.appliedExpUs ?: -1,
                base[0].maxDn.toInt(), base[0].maxDn.toInt(), p, "", "", ""))
            val brightCell = grid.cell(grid.nGain - 1, grid.nShutter - 1)
            runCatching {
                val bmp = parallelFormation.formAllCells(base, intArrayOf(brightCell))[0]
                onFrame(bmp, emptyList(), effIso(grid.gains.last()), grid.exposuresUs.last())
            }
        }
        for (pass in listOf("phys1", "phys2")) {
            for (gi in 0 until grid.nGain) for (sj in 0 until grid.nShutter) {
                if (!running) { endPass(); return }
                val cell = grid.cell(gi, sj)
                val frames = raw.capture(grid.exposuresUs[sj], grid.gains[gi], 1)
                if (frames.isEmpty()) continue
                val m = raw.lastMeta.firstOrNull()
                // Effective ISO in the filename for consistency with the JPEG's content;
                // the physical value is recoverable via the manifest's digital_boost.
                val tag = isoExpTag(effIso(grid.gains[gi]), grid.exposuresUs[sj])
                val rawPath = logger.saveRaw("verify_${pass}_${cell}_$tag", frames[0].bayer)
                val bmp = bitmapFromRaw(frames[0])
                val dets = detector.detectBatch(listOf(bmp))[0]
                val imgPath = logger.saveJpeg("verify_${pass}_${cell}_$tag", bmp)
                logger.row("verify", listOf(System.currentTimeMillis(), "static", pass, cell,
                    effIso(grid.gains[gi]), m?.appliedIso ?: -1, m?.appliedExpUs ?: -1,
                    frames[0].maxDn.toInt(), frames[0].maxDn.toInt(),
                    rawPath, imgPath, dets.size, "%.3f".format(sumConf(dets))))
                // Per-box detail for the IoU comparison against runVerifyProbe.
                val detsArr = JSONArray()
                for (d in dets) detsArr.put(detJson(d))
                logger.jsonl("verify_dets", JSONObject().apply {
                    put("method", "verify_$pass"); put("scene", "static")
                    put("cell", cell)
                    put("gain", effIso(grid.gains[gi])); put("exposure_us", grid.exposuresUs[sj])
                    put("dets", detsArr)
                })
                runCatching { onFrame(bmp, dets, effIso(grid.gains[gi]), grid.exposuresUs[sj]) }
                onStatus("Verify $pass cell=$cell ndet=${dets.size}")
            }
        }
        endPass()
    }

    // ---------- VerifyProbe: virtual half of the probing-realism pair ----------

    /** The controller's probe step in isolation: one burst at (baseGain, fastestExposure),
     *  all N cells formed virtually, one batched inference, then a row per cell.
     *
     *  Pair with [runVerify] on the same static scene. The hypothesis is that per-cell pixel
     *  and detector statistics here match the physically-visited ones. */
    fun runVerifyProbe(isGtReference: Boolean, onStatus: (String) -> Unit,
                       onFrame: (Bitmap, List<Detection>, Int, Int) -> Unit = { _, _, _, _ -> }) {
        startPass("verifyprobe", isGtReference, withFrames = false)
        logger.csv("probe", listOf(
            "ts", "cell", "gain", "exposure_us",
            "formation_ms_total", "infer_ms_total", "n_det", "sum_conf", "img_path"))
        val burst = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        if (burst.isEmpty()) { endPass(); return }
        // Every burst RAW is saved so offline tools can reproduce the formation independently.
        val burstTag = isoExpTag(effIso(grid.baseGain), grid.fastestExposureUs)
        burst.forEachIndexed { k, f -> logger.saveRaw("probe_burst_${k}_$burstTag", f.bayer) }

        val allCells = IntArray(grid.nGain * grid.nShutter) { it }
        val tf = now()
        val bitmaps = parallelFormation.formAllCells(burst, allCells)
        val formMs = ms(tf)
        val ti = now()
        val detsAll = detector.detectBatch(bitmaps)
        val inferMs = ms(ti)
        val nowMs = System.currentTimeMillis()
        for (i in allCells.indices) {
            if (!running) break
            val c = allCells[i]; val (gi, sj) = grid.indices(c)
            val dets = detsAll[i]
            val path = logger.saveJpeg("probe_cell${c}_${isoExpTag(effIso(grid.gains[gi]), grid.exposuresUs[sj])}", bitmaps[i])
            logger.row("probe", listOf(nowMs, c, effIso(grid.gains[gi]), grid.exposuresUs[sj],
                "%.1f".format(formMs), "%.1f".format(inferMs),
                dets.size, "%.3f".format(sumConf(dets)), path))
            // Per-box detail for the IoU comparison against runVerify.
            val detsArr = JSONArray()
            for (d in dets) detsArr.put(detJson(d))
            logger.jsonl("probe_dets", JSONObject().apply {
                put("method", "verifyprobe"); put("cell", c)
                put("gain", effIso(grid.gains[gi])); put("exposure_us", grid.exposuresUs[sj])
                put("dets", detsArr)
            })
            runCatching { onFrame(bitmaps[i], dets, effIso(grid.gains[gi]), grid.exposuresUs[sj]) }
            onStatus("Probe cell=$c ndet=${dets.size} sumConf=${"%.2f".format(sumConf(dets))}")
        }
        endPass()
    }

    // ---------- ISO diagnostic ----------

    /** Sweeps ISO at one fixed exposure and logs mean/std of the black-subtracted RAW pixels,
     *  answering whether the sensor really amplifies when SENSOR_SENSITIVITY changes. The
     *  128× span is far too large to confuse with noise, and std should scale too, since
     *  analog gain amplifies read noise along with signal. Where mean stops scaling is the
     *  analog ceiling — beyond it the HAL is applying digital gain that bypasses RAW. */
    fun runIsoDiag(exposureUs: Int = 16000, framesPerIso: Int = 5,
                   isoList: List<Int> = listOf(25, 100, 400, 800, 1600, 3200),
                   onStatus: (String) -> Unit) {
        writeManifest("iso_diag", isGtReference = false, raw.captureWidth, raw.captureHeight)
        logger.csv("iso_diag", listOf("iso_req", "iso_applied", "exp_req_us", "exp_applied_us",
            "frame_idx", "raw_mean", "raw_std", "raw_min", "raw_max", "black_subtracted"))
        running = true
        for (iso in isoList) {
            if (!running) break
            for (idx in 0 until framesPerIso) {
                if (!running) break
                val frames = raw.capture(exposureUs, iso, 1)
                if (frames.isEmpty()) continue
                val meta = raw.lastMeta.firstOrNull()
                val pixels = frames[0].bayer
                val mean = pixels.average()
                val variance = pixels.sumOf { (it - mean) * (it - mean) } / pixels.size
                val std = kotlin.math.sqrt(variance)
                var lo = Int.MAX_VALUE; var hi = Int.MIN_VALUE
                for (p in pixels) { if (p < lo) lo = p; if (p > hi) hi = p }
                logger.row("iso_diag", listOf(
                    iso, meta?.appliedIso ?: -1,
                    exposureUs, (meta?.appliedExpUs ?: -1L),
                    idx, "%.2f".format(mean), "%.2f".format(std), lo, hi, meta?.black ?: -1))
                onStatus("ISO $iso (applied ${meta?.appliedIso}): mean=${"%.0f".format(mean)} std=${"%.0f".format(std)}")
            }
        }
        logger.flush()
        onStatus("ISO diag done — see iso_diag.csv")
    }

    // ---------- Bench (latency vs K) ----------

    /** One paired visual/detection ablation. Both arms use the exact same captured RAW burst
     *  and centered shutter windows; only AWB+CCM differs. Saves lossless candidates, overlays,
     *  complete boxes/classes/confidences, and a four-class presence summary. */
    fun runAbVisual(onStatus: (String) -> Unit) {
        val targets = linkedMapOf("bus" to 5, "wine_glass" to 40, "cup" to 41, "banana" to 46)
        writeManifest("ab_visual", isGtReference = false, captureWidth = raw.captureWidth,
            captureHeight = raw.captureHeight, methodParams = JSONObject()
                .put("same_raw_for_A_B", true)
                .put("burst_window_both_arms", "centered")
                .put("target_classes", JSONObject(targets as Map<*, *>)))
        logger.csv("ab_visual", listOf(
            "pipeline", "cell", "effective_iso", "exposure_us", "selected",
            "n_det", "sum_conf", "bus_conf", "wine_glass_conf", "cup_conf",
            "banana_conf", "candidate_path", "overlay_path"))
        running = true
        onStatus("A/B Visual · capturing one shared RAW burst")
        val frames = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        if (frames.isEmpty()) { logger.flush(); return }
        frames.forEachIndexed { i, frame -> logger.saveRaw("ab_visual_input_$i", frame.bayer) }
        val metaJson = JSONArray()
        raw.lastMeta.forEach { m -> metaJson.put(JSONObject().apply {
            put("frame_number", m.frameNumber); put("sensor_timestamp", m.timestamp)
            put("requested_iso", m.requestedIso); put("applied_iso", m.appliedIso)
            put("requested_exp_us", m.requestedExpUs); put("applied_exp_us", m.appliedExpUs)
            put("black_levels", JSONArray(m.blackLevels.toTypedArray()))
            put("white_balance", JSONArray(m.whiteBalance.toTypedArray()))
            put("camera_to_srgb", JSONArray(m.cameraToSrgb.toTypedArray()))
        }) }
        File(logger.dir, "ab_visual_capture.json").writeText(JSONObject().apply {
            put("frame_count", frames.size); put("width", frames[0].width)
            put("height", frames[0].height); put("cfa", frames[0].cfaPattern)
            put("white_level", frames[0].maxDn); put("sensor_orientation", frames[0].sensorOrientation)
            put("metadata", metaJson)
        }.toString(2))

        val sources = linkedMapOf(
            ColorPipeline.RAYNEO_AWB_CCM to ParallelRawCandidateSource(
                grid, raw, colorPipeline = ColorPipeline.RAYNEO_AWB_CCM,
                burstWindow = BurstWindow.CENTERED),
            ColorPipeline.ORIGINAL_GAIN_SRGB to ParallelRawCandidateSource(
                grid, raw, colorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB,
                burstWindow = BurstWindow.CENTERED)
        )
        val cells = IntArray(grid.nGain * grid.nShutter) { it }
        val summaries = JSONObject()
        try {
            for ((mode, source) in sources) {
                if (!running) break
                onStatus("A/B Visual · forming ${mode.tag}")
                val images = source.formAllCells(frames, cells)
                val detsByCell = detector.detectBatch(images)
                val scores = detsByCell.map { sumConf(it) }
                val selectedIndex = scores.indices.maxByOrNull { scores[it] } ?: 0
                val selectedCell = cells[selectedIndex]
                for (i in cells.indices) {
                    val cell = cells[i]; val (gi, sj) = grid.indices(cell)
                    val dets = detsByCell[i]
                    val above = dets.filter { it.confidence >= selectConf }
                    fun targetConf(cls: Int): Double = above.asSequence()
                        .filter { it.classId == cls }.maxOfOrNull { it.confidence.toDouble() } ?: 0.0
                    val base = "${mode.tag}_cell${cell}_${isoExpTag(effIso(grid.gains[gi]), grid.exposuresUs[sj])}"
                    val candidatePath = logger.savePng(base, images[i])
                    val overlay = drawDetectionOverlay(images[i], above)
                    val overlayPath = logger.savePng("${base}_boxes", overlay)
                    overlay.recycle()
                    logger.row("ab_visual", listOf(
                        mode.tag, cell, effIso(grid.gains[gi]), grid.exposuresUs[sj],
                        if (cell == selectedCell) 1 else 0, above.size,
                        "%.5f".format(sumConf(above)),
                        "%.5f".format(targetConf(5)), "%.5f".format(targetConf(40)),
                        "%.5f".format(targetConf(41)), "%.5f".format(targetConf(46)),
                        candidatePath, overlayPath))
                    val arr = JSONArray(); dets.forEach { arr.put(detJson(it)) }
                    logger.jsonl("ab_visual_dets", JSONObject().apply {
                        put("pipeline", mode.tag); put("cell", cell)
                        put("selected", cell == selectedCell); put("dets", arr)
                    })
                }
                val selectedAbove = detsByCell[selectedIndex].filter { it.confidence >= selectConf }
                val targetResult = JSONObject()
                targets.forEach { (name, cls) -> targetResult.put(name,
                    selectedAbove.filter { it.classId == cls }.maxOfOrNull { it.confidence.toDouble() } ?: 0.0) }
                summaries.put(mode.tag, JSONObject().apply {
                    put("selected_cell", selectedCell); put("selected_sum_conf", scores[selectedIndex])
                    put("target_confidence", targetResult)
                    put("target_recall", targets.values.count { cls -> selectedAbove.any { it.classId == cls } } / 4.0)
                    put("detections_at_threshold", JSONArray(selectedAbove.map { detJson(it) }))
                })
                images.forEach { if (!it.isRecycled) it.recycle() }
            }
        } finally {
            sources.values.forEach { it.shutdown() }
        }
        File(logger.dir, "ab_visual_summary.json").writeText(summaries.toString(2))
        logger.flush()
        onStatus("A/B Visual done · saved candidates, boxes and target summary")
    }

    private fun drawDetectionOverlay(source: Bitmap, dets: List<Detection>): Bitmap {
        val out = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val box = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 4f
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.GREEN; style = Paint.Style.FILL; textSize = 24f
        }
        for (d in dets) {
            val p = d.xyxy
            canvas.drawRect(p[0], p[1], p[2], p[3], box)
            canvas.drawText("${CocoLabels.name(d.classId)} %.2f".format(d.confidence),
                p[0].coerceAtLeast(2f), (p[1] - 6f).coerceAtLeast(24f), textPaint)
        }
        return out
    }

    /** Paired A/B benchmark over one immutable RAW burst. Every measured call includes
     *  candidate formation, YOLO preprocessing, GPU execution and decode, but excludes the
     *  one-time capture and UI rendering. Alternating order limits thermal/order bias. */
    fun runAbBench(onStatus: (String) -> Unit, warmups: Int = 5, repetitions: Int = 20) {
        writeManifest("ab_bench", isGtReference = false, captureWidth = raw.captureWidth,
            captureHeight = raw.captureHeight, methodParams = JSONObject()
                .put("warmups", warmups).put("repetitions", repetitions)
                .put("same_raw_for_all_conditions", true)
                .put("deployed_pipeline", colorPipeline.tag))
        logger.csv("ab_bench", listOf(
            "phase", "rep", "pipeline", "k", "distinct_rows",
            "row_wall_ms", "sum_cpu_ms", "demosaic_cpu_ms",
            "pack_wall_ms", "pack_cpu_ms", "bitmap_ms", "formation_ms",
            "yolo_preprocess_ms", "gpu_run_ms", "decode_ms", "infer_total_ms",
            "pipeline_total_ms", "n_det", "sum_conf", "battery_temp_c",
            "thermal_status", "pss_kb"))
        running = true
        onStatus("A/B capture — one RAW burst for every condition")
        val captureStart = now()
        val frames = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        val captureMs = ms(captureStart)
        if (frames.isEmpty()) { logger.flush(); return }
        frames.forEachIndexed { i, frame -> logger.saveRaw("ab_input_$i", frame.bayer) }
        File(logger.dir, "ab_capture.json").writeText(JSONObject().apply {
            put("capture_ms", captureMs)
            put("frame_count", frames.size)
            put("width", frames[0].width); put("height", frames[0].height)
            put("raw_metadata_matched_by", "SENSOR_TIMESTAMP")
            put("black_level", "per-CFA dynamic with static fallback")
        }.toString(2))

        val sources = linkedMapOf(
            ColorPipeline.RAYNEO_AWB_CCM to ParallelRawCandidateSource(
                grid, raw, colorPipeline = ColorPipeline.RAYNEO_AWB_CCM,
                burstWindow = BurstWindow.CENTERED),
            ColorPipeline.ORIGINAL_GAIN_SRGB to ParallelRawCandidateSource(
                grid, raw, colorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB,
                burstWindow = BurstWindow.CENTERED)
        )
        val cellSets = linkedMapOf(
            1 to intArrayOf(grid.cell(0, 0)),
            grid.nGain to IntArray(grid.nGain) { grid.cell(it, 0) },
            grid.nGain * grid.nShutter to IntArray(grid.nGain * grid.nShutter) { it }
        )
        try {
            val totalReps = warmups + repetitions
            for (rep in 0 until totalReps) {
                if (!running) break
                val kOrder = if (rep % 2 == 0) cellSets.entries.toList()
                    else cellSets.entries.toList().asReversed()
                for ((k, cells) in kOrder) {
                    val modeOrder = if ((rep + k) % 2 == 0) sources.entries.toList()
                        else sources.entries.toList().asReversed()
                    for ((mode, source) in modeOrder) {
                        if (!running) break
                        val totalStart = now()
                        val images = source.formAllCells(frames, cells)
                        val inferStart = now()
                        val detections = detector.detectBatch(images)
                        val inferMs = ms(inferStart)
                        val totalMs = ms(totalStart)
                        val profile = requireNotNull(source.lastProfile)
                        val yolo = detector as? com.example.activeperception.acquire.TfliteYoloDetector
                        val healthNow = health.sample()
                        val allDets = detections.flatten()
                        logger.row("ab_bench", listOf(
                            if (rep < warmups) "warmup" else "measure",
                            if (rep < warmups) rep else rep - warmups,
                            mode.tag, k, profile.distinctRows,
                            "%.3f".format(profile.rowWallMs),
                            "%.3f".format(profile.sumCpuMs),
                            "%.3f".format(profile.demosaicCpuMs),
                            "%.3f".format(profile.packWallMs),
                            "%.3f".format(profile.packCpuMs),
                            "%.3f".format(profile.bitmapMs),
                            "%.3f".format(profile.totalMs),
                            "%.3f".format(yolo?.lastPreprocessMs ?: -1.0),
                            "%.3f".format(yolo?.lastRunMs ?: -1.0),
                            "%.3f".format(yolo?.lastDecodeMs ?: -1.0),
                            "%.3f".format(inferMs), "%.3f".format(totalMs),
                            allDets.size, "%.5f".format(sumConf(allDets)),
                            healthNow.batteryTemperatureC, healthNow.thermalStatus,
                            healthNow.totalPssKb))
                        images.forEach { if (!it.isRecycled) it.recycle() }
                    }
                }
                onStatus("A/B ${rep + 1}/$totalReps · paired B1/B3/B9")
            }
        } finally {
            sources.values.forEach { it.shutdown() }
        }
        logger.flush()
        onStatus("A/B done — capture=${"%.0f".format(captureMs)}ms")
    }

    /** EXP2.1 | RGB -> YOLO Tensor direct generation.
     *
     *  C keeps the deployed Bitmap path; D writes the model tensor directly. Both arms share
     *  one immutable RAW burst and one prepared set of demosaiced RGB planes. This is an
     *  experiment-only route: normal Fixed/AE/Proposed execution is not changed. */
    fun runExp21DirectTensor(onStatus: (String) -> Unit, warmups: Int = 5, repetitions: Int = 20) {
        val yolo = detector as? com.example.activeperception.acquire.TfliteYoloDetector
            ?: error("EXP2.1 requires TfliteYoloDetector")
        writeManifest("exp2_1_rgb_direct_tensor", isGtReference = false,
            captureWidth = raw.captureWidth, captureHeight = raw.captureHeight,
            methodParams = JSONObject()
                .put("title", "EXP2.1 | RGB -> YOLO Tensor direct generation")
                .put("control", "C_bitmap")
                .put("treatment", "E_native_neon")
                .put("native_impl", "C++17 + ARM NEON + striped CPU parallelism")
                .put("shared_raw", true).put("shared_demosaiced_rgb", true)
                .put("warmups", warmups).put("repetitions", repetitions)
                .put("production_pipeline_changed", false))
        logger.csv("exp21", listOf(
            "phase", "rep", "path", "k", "rgb_prepare_ms",
            "transform_wall_ms", "transform_cpu_ms", "bitmap_create_ms",
            "yolo_preprocess_ms", "gpu_run_ms", "decode_ms", "infer_total_ms",
            "detection_ready_ms", "selected_display_ms", "display_ready_ms",
            "n_det", "sum_conf", "battery_temp_c", "thermal_status", "pss_kb"))
        logger.csv("exp21_equivalence", listOf(
            "k", "bitmap_n", "direct_n", "matched_iou50", "reference_n",
            "mean_matched_iou", "mean_conf_abs_delta", "class_set_equal_lanes",
            "tensor_mae_k1", "tensor_rmse_k1", "tensor_max_abs_k1"))

        running = true
        onStatus("EXP2.1 capture — immutable RAW burst")
        val captureStart = now()
        val frames = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        val captureMs = ms(captureStart)
        if (frames.isEmpty()) { logger.flush(); return }
        frames.forEachIndexed { i, frame -> logger.saveRaw("exp21_input_$i", frame.bayer) }

        val source = ParallelRawCandidateSource(grid, raw,
            colorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB,
            burstWindow = BurstWindow.FIRST_N)
        val cellSets = linkedMapOf(
            1 to intArrayOf(grid.cell(0, 0)),
            grid.nGain to IntArray(grid.nGain) { grid.cell(it, 0) },
            grid.nGain * grid.nShutter to IntArray(grid.nGain * grid.nShutter) { it })

        fun iou(a: FloatArray, b: FloatArray): Double {
            val x1 = maxOf(a[0], b[0]); val y1 = maxOf(a[1], b[1])
            val x2 = minOf(a[2], b[2]); val y2 = minOf(a[3], b[3])
            val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
            val aa = maxOf(0f, a[2] - a[0]) * maxOf(0f, a[3] - a[1])
            val bb = maxOf(0f, b[2] - b[0]) * maxOf(0f, b[3] - b[1])
            val union = aa + bb - inter
            return if (union > 0f) inter / union.toDouble() else 0.0
        }

        try {
            for ((k, cells) in cellSets) {
                if (!running) break
                onStatus("EXP2.1 preparing shared RGB K=$k")
                val prepared = source.prepareExp21Rgb(frames, cells)

                // Untimed one-off equivalence probe.
                val control = source.formExp21Bitmaps(prepared)
                val controlDets = yolo.detectBatch(control.images)
                val direct = source.formExp21NativeTensor(prepared)
                val directDets = yolo.detectTensorBatch(direct.batch)
                var referenceN = 0; var matched = 0; var iouSum = 0.0; var confDelta = 0.0
                var equalClassLanes = 0
                for (lane in 0 until k) {
                    val c = controlDets[lane].filter { it.confidence >= selectConf }
                    val d = directDets[lane].filter { it.confidence >= selectConf }
                    if (c.map { it.classId }.toSet() == d.map { it.classId }.toSet()) equalClassLanes++
                    for (box in c) {
                        referenceN++
                        val best = d.filter { it.classId == box.classId }
                            .maxByOrNull { iou(box.xyxy, it.xyxy) }
                        if (best != null) {
                            val overlap = iou(box.xyxy, best.xyxy)
                            if (overlap >= 0.5) {
                                matched++; iouSum += overlap
                                confDelta += kotlin.math.abs(box.confidence - best.confidence)
                            }
                        }
                    }
                }
                var tensorMae = -1.0; var tensorRmse = -1.0; var tensorMax = -1.0
                if (k == 1) {
                    val snapshot = yolo.snapshotBitmapTensor(control.images)
                    var absSum = 0.0; var sqSum = 0.0; var maxAbs = 0.0
                    for (i in snapshot.values.indices) {
                        val delta = kotlin.math.abs(snapshot.values[i] - direct.batch.input.getFloat(i * 4)).toDouble()
                        absSum += delta; sqSum += delta * delta; if (delta > maxAbs) maxAbs = delta
                    }
                    tensorMae = absSum / snapshot.values.size
                    tensorRmse = kotlin.math.sqrt(sqSum / snapshot.values.size)
                    tensorMax = maxAbs
                    fun tensorBitmap(read: (Int) -> Float): Bitmap {
                        val pixels = IntArray(640 * 640)
                        for (p in pixels.indices) {
                            val base = p * 3
                            val r = (read(base) * 255f).toInt().coerceIn(0, 255)
                            val g = (read(base + 1) * 255f).toInt().coerceIn(0, 255)
                            val b = (read(base + 2) * 255f).toInt().coerceIn(0, 255)
                            pixels[p] = Color.argb(255, r, g, b)
                        }
                        return Bitmap.createBitmap(pixels, 640, 640, Bitmap.Config.ARGB_8888)
                    }
                    val cTensor = tensorBitmap { snapshot.values[it] }
                    val dTensor = tensorBitmap { direct.batch.input.getFloat(it * 4) }
                    logger.savePng("exp21_C_bitmap_tensor_k1", cTensor)
                    logger.savePng("exp21_E_native_neon_tensor_k1", dTensor)
                    cTensor.recycle(); dTensor.recycle()
                }
                logger.row("exp21_equivalence", listOf(k,
                    controlDets.sumOf { it.count { d -> d.confidence >= selectConf } },
                    directDets.sumOf { it.count { d -> d.confidence >= selectConf } },
                    matched, referenceN,
                    "%.6f".format(if (matched > 0) iouSum / matched else 0.0),
                    "%.6f".format(if (matched > 0) confDelta / matched else 0.0),
                    equalClassLanes,
                    "%.8f".format(tensorMae), "%.8f".format(tensorRmse),
                    "%.8f".format(tensorMax)))
                logger.jsonl("exp21_dets", JSONObject().apply {
                    put("k", k)
                    put("C_bitmap", JSONArray(controlDets.map { lane -> JSONArray(lane.map { detJson(it) }) }))
                    put("E_native_neon", JSONArray(directDets.map { lane -> JSONArray(lane.map { detJson(it) }) }))
                })
                control.images.forEach { if (!it.isRecycled) it.recycle() }

                val totalReps = warmups + repetitions
                for (rep in 0 until totalReps) {
                    if (!running) break
                    val paths = if (rep % 2 == 0) listOf("C_bitmap", "E_native_neon")
                        else listOf("E_native_neon", "C_bitmap")
                    for (path in paths) {
                        val profile: Exp21PathProfile
                        val detections: List<List<Detection>>
                        val inferMs: Double
                        var selectedDisplayMs = 0.0
                        if (path == "C_bitmap") {
                            val formed = source.formExp21Bitmaps(prepared)
                            profile = formed.profile
                            val inferStart = now(); detections = yolo.detectBatch(formed.images)
                            inferMs = ms(inferStart)
                            formed.images.forEach { if (!it.isRecycled) it.recycle() }
                        } else {
                            val formed = source.formExp21NativeTensor(prepared)
                            profile = formed.profile
                            val inferStart = now(); detections = yolo.detectTensorBatch(formed.batch)
                            inferMs = ms(inferStart)
                            val selectedLane = detections.indices.maxByOrNull { lane ->
                                sumConf(detections[lane].filter { it.confidence >= selectConf })
                            } ?: 0
                            val displayPrepared = prepared.copy(
                                cells = intArrayOf(prepared.cells[selectedLane]))
                            val display = source.formExp21Bitmaps(displayPrepared)
                            selectedDisplayMs = display.profile.totalMs
                            display.images.forEach { if (!it.isRecycled) it.recycle() }
                        }
                        val detectionReadyMs = profile.totalMs + inferMs
                        val displayReadyMs = detectionReadyMs + selectedDisplayMs
                        val flat = detections.flatten().filter { it.confidence >= selectConf }
                        val h = health.sample()
                        logger.row("exp21", listOf(
                            if (rep < warmups) "warmup" else "measure",
                            if (rep < warmups) rep else rep - warmups,
                            path, k, "%.3f".format(prepared.prepareMs),
                            "%.3f".format(profile.transformWallMs),
                            "%.3f".format(profile.transformCpuMs),
                            "%.3f".format(profile.bitmapMs),
                            "%.3f".format(yolo.lastPreprocessMs),
                            "%.3f".format(yolo.lastRunMs),
                            "%.3f".format(yolo.lastDecodeMs),
                            "%.3f".format(inferMs), "%.3f".format(detectionReadyMs),
                            "%.3f".format(selectedDisplayMs), "%.3f".format(displayReadyMs),
                            flat.size, "%.5f".format(sumConf(flat)),
                            h.batteryTemperatureC, h.thermalStatus, h.totalPssKb))
                    }
                    onStatus("EXP2.1 K=$k ${rep + 1}/$totalReps")
                }
            }
        } finally {
            source.shutdown()
        }
        File(logger.dir, "exp21_capture.json").writeText(JSONObject()
            .put("capture_ms", captureMs).put("frames", frames.size)
            .put("excluded_from_exp21_latency", true).toString(2))
        logger.flush()
        onStatus("EXP2.1 done")
    }

    /** EXP2.2 | YOLO output decode/NMS optimization.
     *
     * A and B use the same immutable RAW burst, prepared RGB, native tensor generation,
     * TFLite model, batch interpreter, and GPU delegate. Only the CPU output decoder changes:
     * A is the deployed flatten/object decoder; B is direct-output + reusable primitive scratch
     * + pre-NMS Top-K + primitive NMS + parallel batch-lane decode. Production remains A. */
    fun runExp22DecodeOptimization(
        onStatus: (String) -> Unit, warmups: Int = 5, repetitions: Int = 20
    ) {
        val yolo = detector as? com.example.activeperception.acquire.TfliteYoloDetector
            ?: error("EXP2.2 requires TfliteYoloDetector")
        writeManifest("exp2_2_decode_nms_optimization", isGtReference = false,
            captureWidth = raw.captureWidth, captureHeight = raw.captureHeight,
            methodParams = JSONObject()
                .put("title", "EXP2.2 | Decode/NMS optimization")
                .put("control", "A_current_decode")
                .put("treatment", "B_optimized_decode")
                .put("shared_raw_rgb_tensor_model_gpu", true)
                .put("optimized_features", JSONArray(listOf(
                    "no_full_output_copy", "decode_array_reuse", "pre_nms_topk_1000",
                    "primitive_nms", "parallel_batch_lane_decode")))
                .put("warmups", warmups).put("repetitions", repetitions)
                .put("production_pipeline_changed", false))
        logger.csv("exp22", listOf(
            "phase", "rep", "path", "k", "tensor_transform_ms", "gpu_run_ms",
            "decode_ms", "gpu_decode_ms", "detection_ready_ms", "pre_nms_candidates",
            "topk_candidates", "n_det_floor", "n_det_select", "sum_conf_select",
            "battery_temp_c", "thermal_status", "pss_kb"))
        logger.csv("exp22_equivalence", listOf(
            "k", "threshold", "baseline_n", "optimized_n", "matched_iou50",
            "mean_matched_iou", "mean_conf_abs_delta", "class_set_equal_lanes",
            "missing_baseline", "extra_optimized"))

        running = true
        onStatus("EXP2.2 capture — immutable RAW burst")
        val captureStart = now()
        val frames = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        val captureMs = ms(captureStart)
        if (frames.isEmpty()) { logger.flush(); return }

        val source = ParallelRawCandidateSource(grid, raw,
            colorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB,
            burstWindow = BurstWindow.FIRST_N)
        val cellSets = linkedMapOf(
            1 to intArrayOf(grid.cell(0, 0)),
            grid.nGain to IntArray(grid.nGain) { grid.cell(it, 0) },
            grid.nGain * grid.nShutter to IntArray(grid.nGain * grid.nShutter) { it })

        fun overlap(a: FloatArray, b: FloatArray): Double {
            val x1 = maxOf(a[0], b[0]); val y1 = maxOf(a[1], b[1])
            val x2 = minOf(a[2], b[2]); val y2 = minOf(a[3], b[3])
            val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
            val aa = maxOf(0f, a[2] - a[0]) * maxOf(0f, a[3] - a[1])
            val bb = maxOf(0f, b[2] - b[0]) * maxOf(0f, b[3] - b[1])
            val union = aa + bb - inter
            return if (union > 0f) inter / union.toDouble() else 0.0
        }

        fun logEquivalence(k: Int, threshold: Float,
                           baseline: List<List<Detection>>, optimized: List<List<Detection>>) {
            var baselineN = 0; var optimizedN = 0; var matched = 0
            var iouSum = 0.0; var confSum = 0.0; var equalLanes = 0
            for (lane in 0 until k) {
                val a = baseline[lane].filter { it.confidence >= threshold }
                val b = optimized[lane].filter { it.confidence >= threshold }
                baselineN += a.size; optimizedN += b.size
                if (a.map { it.classId }.toSet() == b.map { it.classId }.toSet()) equalLanes++
                val used = BooleanArray(b.size)
                for (reference in a.sortedByDescending { it.confidence }) {
                    var bestIndex = -1; var bestIou = -1.0
                    for (i in b.indices) {
                        if (used[i] || b[i].classId != reference.classId) continue
                        val iou = overlap(reference.xyxy, b[i].xyxy)
                        if (iou > bestIou) { bestIou = iou; bestIndex = i }
                    }
                    if (bestIndex >= 0 && bestIou >= 0.5) {
                        used[bestIndex] = true; matched++; iouSum += bestIou
                        confSum += kotlin.math.abs(reference.confidence - b[bestIndex].confidence)
                    }
                }
            }
            logger.row("exp22_equivalence", listOf(k, threshold,
                baselineN, optimizedN, matched,
                "%.6f".format(if (matched > 0) iouSum / matched else 0.0),
                "%.8f".format(if (matched > 0) confSum / matched else 0.0),
                equalLanes, baselineN - matched, optimizedN - matched))
        }

        try {
            for ((k, cells) in cellSets) {
                if (!running) break
                onStatus("EXP2.2 preparing shared tensor K=$k")
                val prepared = source.prepareExp21Rgb(frames, cells)
                val probe = source.formExp21NativeTensor(prepared)
                val baseline = yolo.detectTensorBatch(probe.batch)
                val optimized = yolo.detectTensorBatchOptimized(probe.batch)
                logEquivalence(k, 0.01f, baseline, optimized)
                logEquivalence(k, selectConf, baseline, optimized)
                logger.jsonl("exp22_dets", JSONObject().apply {
                    put("k", k)
                    put("A_current_decode", JSONArray(baseline.map { lane ->
                        JSONArray(lane.map { detJson(it) }) }))
                    put("B_optimized_decode", JSONArray(optimized.map { lane ->
                        JSONArray(lane.map { detJson(it) }) }))
                })

                val totalReps = warmups + repetitions
                for (rep in 0 until totalReps) {
                    if (!running) break
                    val paths = if (rep % 2 == 0)
                        listOf("A_current_decode", "B_optimized_decode")
                    else listOf("B_optimized_decode", "A_current_decode")
                    for (path in paths) {
                        val formed = source.formExp21NativeTensor(prepared)
                        val inferStart = now()
                        val detections = if (path == "A_current_decode")
                            yolo.detectTensorBatch(formed.batch)
                        else yolo.detectTensorBatchOptimized(formed.batch)
                        val inferMs = ms(inferStart)
                        val flatFloor = detections.flatten()
                        val flatSelect = flatFloor.filter { it.confidence >= selectConf }
                        val h = health.sample()
                        logger.row("exp22", listOf(
                            if (rep < warmups) "warmup" else "measure",
                            if (rep < warmups) rep else rep - warmups,
                            path, k, "%.3f".format(formed.profile.totalMs),
                            "%.3f".format(yolo.lastRunMs), "%.3f".format(yolo.lastDecodeMs),
                            "%.3f".format(inferMs),
                            "%.3f".format(formed.profile.totalMs + inferMs),
                            if (path == "B_optimized_decode") yolo.lastPreNmsCandidates else -1,
                            if (path == "B_optimized_decode") yolo.lastTopKCandidates else -1,
                            flatFloor.size, flatSelect.size, "%.5f".format(sumConf(flatSelect)),
                            h.batteryTemperatureC, h.thermalStatus, h.totalPssKb))
                    }
                    onStatus("EXP2.2 K=$k ${rep + 1}/$totalReps")
                }
            }
        } finally {
            source.shutdown()
        }
        File(logger.dir, "exp22_capture.json").writeText(JSONObject()
            .put("capture_ms", captureMs).put("frames", frames.size)
            .put("excluded_from_exp22_latency", true).toString(2))
        logger.flush()
        onStatus("EXP2.2 done")
    }

    /** EXP2.3 | Five-class decoder restriction vs five-output model head.
     *
     * Both arms use the EXP2.2 optimized decoder and the same five COCO classes. Arm A keeps
     * the [B,84,8400] All-COCO model and scans five class channels. Arm B uses a head-pruned
     * [B,9,8400] model whose local outputs map to the same COCO IDs. */
    fun runExp23Coco5Comparison(
        onStatus: (String) -> Unit, warmups: Int = 5, repetitions: Int = 20
    ) {
        val allCoco = detector as? com.example.activeperception.acquire.TfliteYoloDetector
            ?: error("EXP2.3 requires TfliteYoloDetector")
        val cocoIds = com.example.activeperception.acquire.TfliteYoloDetector.COCO5_CLASS_IDS
        val cocoNames = listOf("cup", "wine glass", "banana", "bus", "dining table")
        writeManifest("exp2_3_coco5_decoder_vs_head", isGtReference = false,
            captureWidth = raw.captureWidth, captureHeight = raw.captureHeight,
            methodParams = JSONObject()
                .put("title", "EXP2.3 | COCO decoder-5 vs five-output model")
                .put("arm_A", "All-COCO [B,84,8400], decoder scans five channels")
                .put("arm_B", "head-pruned [B,9,8400], five local class channels")
                .put("coco_class_ids", JSONArray(cocoIds.toList()))
                .put("coco_class_names", JSONArray(cocoNames))
                .put("head_pruned_not_retrained", true)
                .put("shared_raw_rgb_tensor", true)
                .put("optimized_exp22_decoder_both_arms", true)
                .put("warmups", warmups).put("repetitions", repetitions))
        logger.csv("exp23", listOf(
            "phase", "rep", "path", "k", "output_channels", "output_bytes",
            "tensor_transform_ms", "gpu_run_ms", "decode_ms", "gpu_decode_ms",
            "detection_ready_ms", "pre_nms_candidates", "topk_candidates",
            "n_det_floor", "n_det_select", "sum_conf_select",
            "battery_temp_c", "thermal_status", "pss_kb"))
        logger.csv("exp23_equivalence", listOf(
            "k", "threshold", "all_coco_n", "head5_n", "matched_iou50",
            "mean_matched_iou", "mean_conf_abs_delta", "class_set_equal_lanes",
            "missing_all_coco", "extra_head5"))
        logger.csv("exp23_model_load", listOf(
            "k", "asset", "load_ms", "backend", "all_coco_output_bytes",
            "head5_output_bytes"))

        running = true
        onStatus("EXP2.3 capture — immutable RAW burst")
        val captureStart = now()
        val frames = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        val captureMs = ms(captureStart)
        if (frames.isEmpty()) { logger.flush(); return }
        val source = ParallelRawCandidateSource(grid, raw,
            colorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB,
            burstWindow = BurstWindow.FIRST_N)
        val cellSets = linkedMapOf(
            1 to intArrayOf(grid.cell(0, 0)),
            grid.nGain to IntArray(grid.nGain) { grid.cell(it, 0) },
            grid.nGain * grid.nShutter to IntArray(grid.nGain * grid.nShutter) { it })
        val assets = mapOf(
            1 to "yolov8n_640_coco5_fp16.tflite",
            3 to "yolov8n_640_b3_coco5_fp16.tflite",
            9 to "yolov8n_640_b9_coco5_fp16.tflite")

        fun overlap(a: FloatArray, b: FloatArray): Double {
            val x1 = maxOf(a[0], b[0]); val y1 = maxOf(a[1], b[1])
            val x2 = minOf(a[2], b[2]); val y2 = minOf(a[3], b[3])
            val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
            val aa = maxOf(0f, a[2] - a[0]) * maxOf(0f, a[3] - a[1])
            val bb = maxOf(0f, b[2] - b[0]) * maxOf(0f, b[3] - b[1])
            val union = aa + bb - inter
            return if (union > 0f) inter / union.toDouble() else 0.0
        }
        fun logEq(k: Int, threshold: Float,
                  a: List<List<Detection>>, b: List<List<Detection>>) {
            var an = 0; var bn = 0; var matched = 0
            var iouSum = 0.0; var confSum = 0.0; var equalLanes = 0
            for (lane in 0 until k) {
                val left = a[lane].filter { it.confidence >= threshold }
                val right = b[lane].filter { it.confidence >= threshold }
                an += left.size; bn += right.size
                if (left.map { it.classId }.toSet() == right.map { it.classId }.toSet()) equalLanes++
                val used = BooleanArray(right.size)
                for (reference in left.sortedByDescending { it.confidence }) {
                    var best = -1; var bestIou = -1.0
                    for (i in right.indices) {
                        if (used[i] || right[i].classId != reference.classId) continue
                        val iou = overlap(reference.xyxy, right[i].xyxy)
                        if (iou > bestIou) { best = i; bestIou = iou }
                    }
                    if (best >= 0 && bestIou >= 0.5) {
                        used[best] = true; matched++; iouSum += bestIou
                        confSum += kotlin.math.abs(reference.confidence - right[best].confidence)
                    }
                }
            }
            logger.row("exp23_equivalence", listOf(k, threshold, an, bn, matched,
                "%.6f".format(if (matched > 0) iouSum / matched else 0.0),
                "%.8f".format(if (matched > 0) confSum / matched else 0.0),
                equalLanes, an - matched, bn - matched))
        }

        try {
            for ((k, cells) in cellSets) {
                if (!running) break
                val asset = requireNotNull(assets[k])
                onStatus("EXP2.3 loading five-output GPU B=$k")
                val loadStart = now()
                val head5 = allCoco.createCoco5HeadDetector(asset, k)
                val loadMs = ms(loadStart)
                logger.row("exp23_model_load", listOf(
                    k, asset, "%.3f".format(loadMs), head5.backendSummary,
                    k * 84 * 8400 * 4, k * 9 * 8400 * 4))
                try {
                    val prepared = source.prepareExp21Rgb(frames, cells)
                    val probe = source.formExp21NativeTensor(prepared)
                    val aProbe = allCoco.detectTensorBatchOptimizedCoco5(probe.batch)
                    val bProbe = head5.detectTensorBatchOptimized(probe.batch)
                    logEq(k, 0.01f, aProbe, bProbe)
                    logEq(k, selectConf, aProbe, bProbe)
                    logger.jsonl("exp23_dets", JSONObject().apply {
                        put("k", k)
                        put("A_all_coco_decoder5", JSONArray(aProbe.map { lane ->
                            JSONArray(lane.map { detJson(it) }) }))
                        put("B_head5_model", JSONArray(bProbe.map { lane ->
                            JSONArray(lane.map { detJson(it) }) }))
                    })

                    val totalReps = warmups + repetitions
                    for (rep in 0 until totalReps) {
                        if (!running) break
                        val paths = if (rep % 2 == 0)
                            listOf("A_all_coco_decoder5", "B_head5_model")
                        else listOf("B_head5_model", "A_all_coco_decoder5")
                        for (path in paths) {
                            val formed = source.formExp21NativeTensor(prepared)
                            val active = if (path == "A_all_coco_decoder5") allCoco else head5
                            val inferStart = now()
                            val detections = if (path == "A_all_coco_decoder5")
                                allCoco.detectTensorBatchOptimizedCoco5(formed.batch)
                            else head5.detectTensorBatchOptimized(formed.batch)
                            val inferMs = ms(inferStart)
                            val floor = detections.flatten()
                            val selected = floor.filter { it.confidence >= selectConf }
                            val h = health.sample()
                            val channels = if (path == "A_all_coco_decoder5") 84 else 9
                            logger.row("exp23", listOf(
                                if (rep < warmups) "warmup" else "measure",
                                if (rep < warmups) rep else rep - warmups,
                                path, k, channels, k * channels * 8400 * 4,
                                "%.3f".format(formed.profile.totalMs),
                                "%.3f".format(active.lastRunMs),
                                "%.3f".format(active.lastDecodeMs),
                                "%.3f".format(inferMs),
                                "%.3f".format(formed.profile.totalMs + inferMs),
                                active.lastPreNmsCandidates, active.lastTopKCandidates,
                                floor.size, selected.size, "%.5f".format(sumConf(selected)),
                                h.batteryTemperatureC, h.thermalStatus, h.totalPssKb))
                        }
                        onStatus("EXP2.3 K=$k ${rep + 1}/$totalReps")
                    }
                } finally {
                    head5.close()
                }
            }
        } finally {
            source.shutdown()
        }
        File(logger.dir, "exp23_capture.json").writeText(JSONObject()
            .put("capture_ms", captureMs).put("frames", frames.size)
            .put("excluded_from_exp23_latency", true).toString(2))
        logger.flush()
        onStatus("EXP2.3 done")
    }

    /** EXP3 | Fully integrated optimized pipeline vs the original EXP1-A baseline.
     *
     * A: RayNeo AWB+CCM -> ARGB/Bitmap -> All-COCO GPU -> original decoder/NMS.
     * B: original gain+sRGB -> native RGB-to-tensor -> five-output GPU -> EXP2.2 decoder/NMS.
     * Both start from the same immutable RAW burst and include sum+demosaic on every call. */
    fun runExp3IntegratedVsExp1A(
        onStatus: (String) -> Unit, warmups: Int = 5, repetitions: Int = 20
    ) {
        val allCoco = detector as? com.example.activeperception.acquire.TfliteYoloDetector
            ?: error("EXP3 requires TfliteYoloDetector")
        val targetIds = com.example.activeperception.acquire.TfliteYoloDetector.COCO5_CLASS_IDS.toSet()
        writeManifest("exp3_integrated_vs_exp1a", isGtReference = false,
            captureWidth = raw.captureWidth, captureHeight = raw.captureHeight,
            methodParams = JSONObject()
                .put("title", "EXP3 | Integrated optimization vs EXP1-A baseline")
                .put("baseline", "AWB+CCM, Bitmap, All-COCO, original decode/NMS")
                .put("optimized", "gain+sRGB, native direct tensor, head5, optimized decode/NMS")
                .put("shared_immutable_raw", true)
                .put("burst_window_both", "CENTERED")
                .put("capture_excluded", true)
                .put("sum_and_demosaic_included_each_repetition", true)
                .put("warmups", warmups).put("repetitions", repetitions))
        logger.csv("exp3", listOf(
            "phase", "rep", "path", "k", "color_pipeline", "output_classes",
            "raw_rgb_prepare_ms", "representation_ms", "formation_total_ms",
            "yolo_preprocess_ms", "gpu_run_ms", "decode_ms", "infer_total_ms",
            "detection_ready_ms", "selected_bitmap_ms", "overlay_ms", "display_ready_ms",
            "n_det_floor", "n_target_select", "sum_conf_target", "selected_lane",
            "battery_temp_c", "thermal_status", "pss_kb"))
        logger.csv("exp3_model_load", listOf("k", "asset", "load_ms", "backend"))

        running = true
        onStatus("EXP3 capture — shared immutable RAW burst")
        val captureStart = now()
        val frames = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        val captureMs = ms(captureStart)
        if (frames.isEmpty()) { logger.flush(); return }
        frames.forEachIndexed { i, frame -> logger.saveRaw("exp3_input_$i", frame.bayer) }
        val baselineSource = ParallelRawCandidateSource(grid, raw,
            colorPipeline = ColorPipeline.RAYNEO_AWB_CCM,
            burstWindow = BurstWindow.CENTERED)
        val optimizedSource = ParallelRawCandidateSource(grid, raw,
            colorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB,
            burstWindow = BurstWindow.CENTERED)
        val cellSets = linkedMapOf(
            1 to intArrayOf(grid.cell(0, 0)),
            grid.nGain to IntArray(grid.nGain) { grid.cell(it, 0) },
            grid.nGain * grid.nShutter to IntArray(grid.nGain * grid.nShutter) { it })
        val assets = mapOf(
            1 to "yolov8n_640_coco5_fp16.tflite",
            3 to "yolov8n_640_b3_coco5_fp16.tflite",
            9 to "yolov8n_640_b9_coco5_fp16.tflite")

        try {
            for ((k, cells) in cellSets) {
                if (!running) break
                val asset = requireNotNull(assets[k])
                onStatus("EXP3 loading optimized head B=$k")
                val loadStart = now()
                val head5 = allCoco.createCoco5HeadDetector(asset, k)
                logger.row("exp3_model_load", listOf(
                    k, asset, "%.3f".format(ms(loadStart)), head5.backendSummary))
                try {
                    val totalReps = warmups + repetitions
                    for (rep in 0 until totalReps) {
                        if (!running) break
                        val paths = if (rep % 2 == 0)
                            listOf("A_EXP1_A_baseline", "B_integrated_optimized")
                        else listOf("B_integrated_optimized", "A_EXP1_A_baseline")
                        for (path in paths) {
                            var rawRgbMs: Double
                            var representationMs: Double
                            val formationMs: Double
                            val detections: List<List<Detection>>
                            val inferMs: Double
                            val active: com.example.activeperception.acquire.TfliteYoloDetector
                            var selectedBitmapMs = 0.0
                            var overlayMs: Double
                            val selectedLane: Int
                            var displayBitmap: Bitmap? = null

                            val totalStart = now()
                            if (path == "A_EXP1_A_baseline") {
                                val images = baselineSource.formAllCells(frames, cells)
                                val profile = requireNotNull(baselineSource.lastProfile)
                                rawRgbMs = profile.rowWallMs
                                representationMs = profile.packWallMs + profile.bitmapMs
                                formationMs = profile.totalMs
                                active = allCoco
                                val inferStart = now()
                                detections = allCoco.detectBatch(images)
                                inferMs = ms(inferStart)
                                selectedLane = detections.indices.maxByOrNull { lane ->
                                    sumConf(detections[lane].filter {
                                        it.classId in targetIds && it.confidence >= selectConf })
                                } ?: 0
                                displayBitmap = images[selectedLane]
                                val overlayStart = now()
                                val overlay = drawDetectionOverlay(displayBitmap,
                                    detections[selectedLane].filter { it.confidence >= selectConf })
                                overlayMs = ms(overlayStart)
                                overlay.recycle()
                                images.forEach { if (!it.isRecycled) it.recycle() }
                            } else {
                                val prepared = optimizedSource.prepareExp21Rgb(frames, cells)
                                rawRgbMs = prepared.prepareMs
                                val formed = optimizedSource.formExp21NativeTensor(prepared)
                                representationMs = formed.profile.totalMs
                                formationMs = prepared.prepareMs + formed.profile.totalMs
                                active = head5
                                val inferStart = now()
                                detections = head5.detectTensorBatchOptimized(formed.batch)
                                inferMs = ms(inferStart)
                                selectedLane = detections.indices.maxByOrNull { lane ->
                                    sumConf(detections[lane].filter { it.confidence >= selectConf })
                                } ?: 0
                                val displayPrepared = prepared.copy(
                                    cells = intArrayOf(prepared.cells[selectedLane]))
                                val bitmapStart = now()
                                val display = optimizedSource.formExp21Bitmaps(displayPrepared)
                                selectedBitmapMs = ms(bitmapStart)
                                displayBitmap = display.images[0]
                                val overlayStart = now()
                                val overlay = drawDetectionOverlay(displayBitmap,
                                    detections[selectedLane].filter { it.confidence >= selectConf })
                                overlayMs = ms(overlayStart)
                                overlay.recycle()
                                display.images.forEach { if (!it.isRecycled) it.recycle() }
                            }
                            val detectionReadyMs = formationMs + inferMs
                            val displayReadyMs = detectionReadyMs + selectedBitmapMs + overlayMs
                            val flat = detections.flatten()
                            val target = flat.filter {
                                it.classId in targetIds && it.confidence >= selectConf }
                            val h = health.sample()
                            logger.row("exp3", listOf(
                                if (rep < warmups) "warmup" else "measure",
                                if (rep < warmups) rep else rep - warmups,
                                path, k,
                                if (path == "A_EXP1_A_baseline") "AWB_CCM" else "GAIN_SRGB",
                                if (path == "A_EXP1_A_baseline") 80 else 5,
                                "%.3f".format(rawRgbMs), "%.3f".format(representationMs),
                                "%.3f".format(formationMs),
                                "%.3f".format(active.lastPreprocessMs),
                                "%.3f".format(active.lastRunMs),
                                "%.3f".format(active.lastDecodeMs),
                                "%.3f".format(inferMs), "%.3f".format(detectionReadyMs),
                                "%.3f".format(selectedBitmapMs), "%.3f".format(overlayMs),
                                "%.3f".format(displayReadyMs), flat.size, target.size,
                                "%.5f".format(sumConf(target)), selectedLane,
                                h.batteryTemperatureC, h.thermalStatus, h.totalPssKb))
                            if (rep == warmups && k == 1) {
                                logger.jsonl("exp3_dets", JSONObject().apply {
                                    put("path", path); put("k", k)
                                    put("detections", JSONArray(detections.map { lane ->
                                        JSONArray(lane.map { detJson(it) }) }))
                                })
                            }
                        }
                        onStatus("EXP3 K=$k ${rep + 1}/$totalReps")
                    }
                } finally {
                    head5.close()
                }
            }
        } finally {
            baselineSource.shutdown(); optimizedSource.shutdown()
        }
        File(logger.dir, "exp3_capture.json").writeText(JSONObject()
            .put("capture_ms", captureMs).put("frames", frames.size)
            .put("excluded_from_exp3_latency", true).toString(2))
        logger.flush()
        onStatus("EXP3 done")
    }

    /** EXP4 | Real P-period controller loop with overlapped, metadata-confirmed Camera2
     * actuation. Unlike EXP3, capture and physical exposure transitions are included.
     *
     * Probe: K=9 at base ISO / fastest exposure. The selected shutter row is requested as
     * soon as inference chooses it. Off-probe: K=3 at that physical exposure. While the
     * current pixels are processed, the next single capture (or return-to-probe burst) is
     * already running on the camera executor. */
    fun runExp4PipelinedPeriods(
        onStatus: (String) -> Unit,
        periods: IntArray = intArrayOf(5),
        cyclesPerCondition: Int = 3
    ) {
        val allCoco = detector as? com.example.activeperception.acquire.TfliteYoloDetector
            ?: error("EXP4 requires TfliteYoloDetector")
        val targetIds = com.example.activeperception.acquire.TfliteYoloDetector.COCO5_CLASS_IDS.toSet()
        writeManifest("exp4_pipelined_periods", isGtReference = false,
            captureWidth = raw.captureWidth, captureHeight = raw.captureHeight,
            methodParams = JSONObject()
                .put("title", "EXP4 | Pipelined physical sensor loop")
                .put("periods", JSONArray(periods.toTypedArray()))
                .put("cycles_per_condition", cyclesPerCondition)
                .put("baseline", "AWB+CCM, Bitmap, All-COCO, original decode/NMS")
                .put("optimized", "gain+sRGB, native tensor, head5, optimized decode/NMS")
                .put("capture_included", true)
                .put("metadata_first_acceptance", true)
                .put("fixed_guard_wait", false)
                .put("capture_processing_overlap", true))
        logger.csv("exp4", listOf(
            "path", "period", "step", "cycle", "phase", "k",
            "requested_exp_us", "applied_exp_us", "applied_iso", "apply_delay_frames",
            "capture_request_to_applied_ms", "capture_wall_ms", "capture_wait_ms",
            "capture_hidden_ms", "formation_ms", "gpu_ms", "decode_ms", "infer_ms",
            "detection_ready_step_ms", "selected_bitmap_ms", "overlay_ms",
            "display_ready_step_ms", "result_interval_ms", "next_request_phase",
            "next_request_queued_after_detection_us", "chosen_cell", "n_target", "sum_conf_target",
            "battery_temp_c", "thermal_status", "pss_kb"))

        data class CapturePacket(
            val requestedExpUs: Int,
            val requestedIso: Int,
            val queuedNs: Long,
            val startedNs: Long,
            val startedSensorClockNs: Long,
            val endedNs: Long,
            val frames: List<RawFrame>,
            val meta: List<RawSensorCapturer.RawMeta>
        )

        val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
        fun submitCapture(exposureUs: Int, nBurst: Int): Future<CapturePacket> {
            val queued = now()
            return cameraExecutor.submit<CapturePacket> {
                val started = now()
                val startedSensorClock = SystemClock.elapsedRealtimeNanos()
                val frames = raw.captureMetadataFirst(exposureUs, grid.baseGain, nBurst)
                val ended = now()
                CapturePacket(exposureUs, grid.baseGain, queued, started, startedSensorClock, ended,
                    frames, raw.lastMeta.map { it.copy(
                        blackLevels = it.blackLevels.copyOf(),
                        whiteBalance = it.whiteBalance.copyOf(),
                        cameraToSrgb = it.cameraToSrgb.copyOf()) })
            }
        }

        val baselineSource = ParallelRawCandidateSource(grid, raw,
            colorPipeline = ColorPipeline.RAYNEO_AWB_CCM,
            burstWindow = BurstWindow.CENTERED)
        val optimizedSource = ParallelRawCandidateSource(grid, raw,
            colorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB,
            burstWindow = BurstWindow.CENTERED)
        var head3: com.example.activeperception.acquire.TfliteYoloDetector? = null
        var head9: com.example.activeperception.acquire.TfliteYoloDetector? = null
        running = true
        try {
            onStatus("EXP4 loading and warming B=3/9")
            head3 = allCoco.createCoco5HeadDetector("yolov8n_640_b3_coco5_fp16.tflite", 3)
            head9 = allCoco.createCoco5HeadDetector("yolov8n_640_b9_coco5_fp16.tflite", 9)
            allCoco.warmUpAllBatches()
            // Alternate arm order by period to reduce monotonic thermal-order bias.
            for ((periodIndex, period) in periods.withIndex()) {
                val paths = if (periodIndex % 2 == 0)
                    listOf("A_EXP1_A_baseline", "B_integrated_optimized")
                else listOf("B_integrated_optimized", "A_EXP1_A_baseline")
                for (path in paths) {
                    if (!running) break
                    val totalSteps = period * cyclesPerCondition
                    var anchor = grid.cell(grid.nGain - 1, 0)
                    var current: Future<CapturePacket> = submitCapture(
                        grid.fastestExposureUs, grid.maxBurst)
                    var previousResultNs = -1L
                    onStatus("EXP4 P=$period $path · $totalSteps steps")
                    for (step in 0 until totalSteps) {
                        if (!running) break
                        val stepStart = now()
                        val isProbe = step % period == 0
                        val cells = if (isProbe) {
                            IntArray(grid.nGain * grid.nShutter) { it }
                        } else {
                            val sj = grid.indices(anchor).second
                            IntArray(grid.nGain) { gi -> grid.cell(gi, sj) }
                        }
                        val waitStart = now()
                        val packet = current.get()
                        val captureWaitMs = ms(waitStart)
                        val captureWallMs = (packet.endedNs - packet.startedNs) / 1e6
                        val captureHiddenMs = (captureWallMs - captureWaitMs).coerceAtLeast(0.0)
                        val meta = packet.meta.firstOrNull()
                        val requestToAppliedMs = meta?.timestamp?.let {
                            (it - packet.startedSensorClockNs).coerceAtLeast(0L) / 1e6
                        } ?: Double.NaN

                        // For every off-probe frame the next physical state is already known.
                        // Queue it before formation/inference, including the single→burst edge.
                        var next: Future<CapturePacket>? = null
                        var nextPhase = "none"
                        var nextQueuedAfterDetectionUs = Double.NaN
                        if (!isProbe && step + 1 < totalSteps) {
                            val nextIsProbe = (step + 1) % period == 0
                            val nextExp = if (nextIsProbe) grid.fastestExposureUs
                                else grid.exposuresUs[grid.indices(anchor).second]
                            val n = if (nextIsProbe) grid.maxBurst else 1
                            next = submitCapture(nextExp, n)
                            nextPhase = if (nextIsProbe) "probe" else "single"
                        }

                        val formationStart = now()
                        val detections: List<List<Detection>>
                        val formationMs: Double
                        val active: com.example.activeperception.acquire.TfliteYoloDetector
                        var displayBitmap: Bitmap? = null
                        var selectedBitmapMs = 0.0
                        var optimizedPrepared: Exp21PreparedRgb? = null
                        var selectedLane = 0
                        if (path == "A_EXP1_A_baseline") {
                            val images = baselineSource.formAllCells(packet.frames, cells)
                            formationMs = ms(formationStart)
                            active = allCoco
                            detections = allCoco.detectBatch(images)
                            selectedLane = detections.indices.maxByOrNull { lane ->
                                sumConf(detections[lane].filter {
                                    it.classId in targetIds && it.confidence >= selectConf })
                            } ?: 0
                            anchor = cells[selectedLane]
                            displayBitmap = images[selectedLane]
                            // Recycle after overlay below; retain only the selected lane now.
                            images.forEachIndexed { i, bmp ->
                                if (i != selectedLane && !bmp.isRecycled) bmp.recycle()
                            }
                        } else {
                            val prepared = optimizedSource.prepareExp21Rgb(packet.frames, cells)
                            optimizedPrepared = prepared
                            val formed = optimizedSource.formExp21NativeTensor(prepared)
                            formationMs = ms(formationStart)
                            active = if (cells.size == 9) head9!! else head3!!
                            detections = active.detectTensorBatchOptimized(formed.batch)
                            selectedLane = detections.indices.maxByOrNull { lane ->
                                sumConf(detections[lane].filter { it.confidence >= selectConf })
                            } ?: 0
                            anchor = cells[selectedLane]
                        }
                        val detectionReadyNs = now()
                        val inferMs = active.lastPreprocessMs + active.lastRunMs + active.lastDecodeMs

                        // Probe→single cannot be predicted. Queue the selected exposure at the
                        // first instruction after selection, before display/overlay/logging.
                        if (isProbe && step + 1 < totalSteps) {
                            val selectedExp = grid.exposuresUs[grid.indices(anchor).second]
                            next = submitCapture(selectedExp, 1)
                            nextPhase = "single"
                            nextQueuedAfterDetectionUs = (now() - detectionReadyNs) / 1e3
                        }

                        if (path == "B_integrated_optimized") {
                            val prepared = requireNotNull(optimizedPrepared)
                            val bitmapStart = now()
                            val one = optimizedSource.formExp21Bitmaps(prepared.copy(
                                cells = intArrayOf(prepared.cells[selectedLane])))
                            selectedBitmapMs = ms(bitmapStart)
                            displayBitmap = one.images[0]
                        }

                        val overlayStart = now()
                        val selectedDets = detections.getOrElse(selectedLane) { emptyList() }
                        val overlay = drawDetectionOverlay(requireNotNull(displayBitmap),
                            selectedDets.filter { it.confidence >= selectConf })
                        val overlayMs = ms(overlayStart)
                        overlay.recycle()
                        displayBitmap.recycle()
                        val resultNs = now()
                        val h = health.sample()
                        val target = selectedDets.filter {
                            it.classId in targetIds && it.confidence >= selectConf
                        }
                        logger.row("exp4", listOf(
                            path, period, step, step / period,
                            if (isProbe) "burst_probe" else "single", cells.size,
                            packet.requestedExpUs, meta?.appliedExpUs ?: -1,
                            meta?.appliedIso ?: -1, meta?.applyDelayFrames ?: -1,
                            "%.3f".format(requestToAppliedMs), "%.3f".format(captureWallMs),
                            "%.3f".format(captureWaitMs), "%.3f".format(captureHiddenMs),
                            "%.3f".format(formationMs), "%.3f".format(active.lastRunMs),
                            "%.3f".format(active.lastDecodeMs), "%.3f".format(inferMs),
                            "%.3f".format((detectionReadyNs - stepStart) / 1e6),
                            "%.3f".format(selectedBitmapMs), "%.3f".format(overlayMs),
                            "%.3f".format((resultNs - stepStart) / 1e6),
                            if (previousResultNs < 0) "" else
                                "%.3f".format((resultNs - previousResultNs) / 1e6),
                            nextPhase,
                            if (nextQueuedAfterDetectionUs.isNaN()) "" else
                                "%.3f".format(nextQueuedAfterDetectionUs),
                            anchor, target.size, "%.5f".format(sumConf(target)),
                            h.batteryTemperatureC, h.thermalStatus, h.totalPssKb))
                        previousResultNs = resultNs
                        if (step + 1 < totalSteps) current = requireNotNull(next)
                        onStatus("EXP4 P=$period ${if (path.startsWith("A")) "A" else "B"} " +
                            "${step + 1}/$totalSteps ${if (isProbe) "burst" else "single"}")
                    }
                }
            }
        } finally {
            running = false
            cameraExecutor.shutdownNow()
            baselineSource.shutdown(); optimizedSource.shutdown()
            runCatching { head3?.close() }; runCatching { head9?.close() }
            logger.flush()
        }
        onStatus("EXP4 done")
    }

    /** EXP5.1 | Camera-only comparison of the current 12+N prequeue against two ways of
     * removing that fixed request window. Formation, inference, display, and file images are
     * deliberately absent so the measured wall time belongs to Camera2 + RAW decode only. */
    fun runExp51CaptureGuardComparison(
        onStatus: (String) -> Unit,
        repeats: Int = 5,
        strategyFilter: String? = null
    ) {
        val exposureSequence = intArrayOf(32_000, 64_000, 128_000, 64_000, 32_000, 128_000)
        val allStrategies = listOf("A_current_12plusN", "B_no_guard_N", "C_adaptive")
        val strategies = strategyFilter?.let { filter ->
            allStrategies.filter { it.startsWith(filter) }
        } ?: allStrategies
        require(strategies.isNotEmpty()) { "Unknown EXP5.1 strategy filter: $strategyFilter" }
        writeManifest("exp5_1_capture_guard", isGtReference = false,
            captureWidth = raw.captureWidth, captureHeight = raw.captureHeight,
            methodParams = JSONObject()
                .put("title", "EXP5.1 | RAW request guard comparison")
                .put("camera_only", true)
                .put("formation_inference_display", false)
                .put("strategies", JSONArray(strategies))
                .put("exposure_sequence_us", JSONArray(exposureSequence.toTypedArray()))
                .put("n_burst", JSONArray(arrayOf(1, 4)))
                .put("repeats", repeats))
        logger.csv("exp51", listOf(
            "strategy", "n_burst", "rep", "seq_idx", "transition",
            "exposure_req_us", "iso_req", "success", "wall_ms",
            "request_to_first_applied_ms", "applied_exp_first_us", "applied_exp_last_us",
            "applied_iso_first", "apply_delay_frames", "returned_frames",
            "all_metadata_match", "sensor_span_ms", "settling_meta_count", "error",
            "battery_temp_c", "thermal_status", "pss_kb"))
        running = true
        try {
            for (nBurst in intArrayOf(1, 4)) {
                for (strategy in strategies) {
                    var previousExp = -1
                    for (rep in 0 until repeats) {
                        for ((seqIdx, expUs) in exposureSequence.withIndex()) {
                            if (!running) break
                            val sensorClockStart = SystemClock.elapsedRealtimeNanos()
                            val wallStart = now()
                            var frames: List<RawFrame> = emptyList()
                            var error = ""
                            try {
                                frames = when (strategy) {
                                    "A_current_12plusN" ->
                                        raw.captureMetadataFirst(expUs, grid.baseGain, nBurst)
                                    "B_no_guard_N" ->
                                        raw.captureNoGuard(expUs, grid.baseGain, nBurst)
                                    else -> raw.captureAdaptive(expUs, grid.baseGain, nBurst)
                                }
                            } catch (t: Throwable) {
                                android.util.Log.e("EXP51", "$strategy N=$nBurst exp=$expUs failed", t)
                                error = (t.message ?: t.javaClass.simpleName)
                                    .replace(',', ';').replace('\n', ' ')
                            }
                            val wallMs = ms(wallStart)
                            val metas = raw.lastMeta
                            val first = metas.firstOrNull()
                            val last = metas.lastOrNull()
                            val requestToApplied = first?.timestamp?.let {
                                (it - sensorClockStart).coerceAtLeast(0L) / 1e6
                            } ?: Double.NaN
                            val allMatch = frames.size == nBurst && metas.size == nBurst &&
                                metas.all {
                                    kotlin.math.abs(it.appliedExpUs - expUs) <= 750L &&
                                        it.appliedIso == grid.baseGain
                                }
                            val sensorSpan = if (first != null && last != null)
                                (last.timestamp - first.timestamp).coerceAtLeast(0L) / 1e6
                            else Double.NaN
                            val h = health.sample()
                            logger.row("exp51", listOf(
                                strategy, nBurst, rep, seqIdx,
                                if (previousExp < 0) "initial" else "${previousExp}_to_$expUs",
                                expUs, grid.baseGain, if (error.isEmpty()) 1 else 0,
                                "%.3f".format(wallMs),
                                if (requestToApplied.isNaN()) "" else "%.3f".format(requestToApplied),
                                first?.appliedExpUs ?: -1, last?.appliedExpUs ?: -1,
                                first?.appliedIso ?: -1, first?.applyDelayFrames ?: -1,
                                frames.size, if (allMatch) 1 else 0,
                                if (sensorSpan.isNaN()) "" else "%.3f".format(sensorSpan),
                                raw.lastSettlingMeta.size, error,
                                h.batteryTemperatureC, h.thermalStatus, h.totalPssKb))
                            previousExp = expUs
                            onStatus("EXP5.1 ${strategy.substring(0, 1)} N=$nBurst " +
                                "${rep * exposureSequence.size + seqIdx + 1}/${repeats * exposureSequence.size}")
                        }
                    }
                }
            }
        } finally {
            running = false
            logger.flush()
        }
        onStatus("EXP5.1 done")
    }

    /** EXP5.1.1 | Fine-grained exact-N capture timing and native RAW decode comparison.
     * Run each [path] in a fresh process/camera session to avoid cross-arm HAL state. */
    fun runExp511NoGuardDecode(
        onStatus: (String) -> Unit,
        path: String,
        repeats: Int = 5
    ) {
        require(path == "B_kotlin" || path == "D_native_neon" || path == "E_native_preview")
        val exposureSequence = intArrayOf(32_000, 64_000, 128_000, 64_000, 32_000, 128_000)
        writeManifest("exp5_1_1_${path.lowercase()}", isGtReference = false,
            captureWidth = raw.captureWidth, captureHeight = raw.captureHeight,
            methodParams = JSONObject()
                .put("title", "EXP5.1.1 | exact-N RAW decode optimization")
                .put("path", path)
                .put("camera_only", true)
                .put("fixed_guard", false)
                .put("exposure_sequence_us", JSONArray(exposureSequence.toTypedArray()))
                .put("n_burst", JSONArray(arrayOf(1, 4)))
                .put("repeats", repeats))
        logger.csv("exp511", listOf(
            "path", "n_burst", "rep", "seq_idx", "exposure_req_us", "success",
            "wall_ms", "first_result_ms", "first_image_ms", "first_matched_ms",
            "decode_wall_ms", "decode_cpu_sum_ms", "cleanup_ms", "profile_total_ms",
            "request_to_applied_ms", "sensor_span_ms", "returned_frames",
            "all_metadata_match", "native_mismatch_count", "sample_checksum", "error",
            "battery_temp_c", "thermal_status", "pss_kb"))
        fun fmt(value: Double): String = if (value.isNaN()) "" else "%.3f".format(value)
        running = true
        try {
            for (nBurst in intArrayOf(1, 4)) {
                // One unmeasured warm-up per capture shape. The native warm-up also performs
                // a full per-pixel equivalence check against the Kotlin reference decoder.
                if (path != "B_kotlin") raw.armNativeDecodeValidation()
                when (path) {
                    "D_native_neon" -> raw.captureNoGuardNative(32_000, grid.baseGain, nBurst)
                    "E_native_preview" -> raw.captureNoGuardNativePreview(32_000, grid.baseGain, nBurst)
                    else -> raw.captureNoGuard(32_000, grid.baseGain, nBurst)
                }
                if (path != "B_kotlin") check(raw.lastNativeDecodeMismatchCount == 0) {
                    "Native RAW decode mismatch count=${raw.lastNativeDecodeMismatchCount}"
                }

                for (rep in 0 until repeats) {
                    for ((seqIdx, expUs) in exposureSequence.withIndex()) {
                        if (!running) break
                        val sensorStart = SystemClock.elapsedRealtimeNanos()
                        val wallStart = now()
                        var frames: List<RawFrame> = emptyList()
                        var error = ""
                        try {
                            frames = when (path) {
                                "D_native_neon" -> raw.captureNoGuardNative(expUs, grid.baseGain, nBurst)
                                "E_native_preview" -> raw.captureNoGuardNativePreview(expUs, grid.baseGain, nBurst)
                                else -> raw.captureNoGuard(expUs, grid.baseGain, nBurst)
                            }
                        } catch (t: Throwable) {
                            android.util.Log.e("EXP511", "$path N=$nBurst exp=$expUs failed", t)
                            error = (t.message ?: t.javaClass.simpleName)
                                .replace(',', ';').replace('\n', ' ')
                        }
                        val wallMs = ms(wallStart)
                        val metas = raw.lastMeta
                        val first = metas.firstOrNull(); val last = metas.lastOrNull()
                        val requestToApplied = first?.timestamp?.let {
                            (it - sensorStart).coerceAtLeast(0L) / 1e6
                        } ?: Double.NaN
                        val sensorSpan = if (first != null && last != null)
                            (last.timestamp - first.timestamp).coerceAtLeast(0L) / 1e6
                        else Double.NaN
                        val allMatch = frames.size == nBurst && metas.size == nBurst &&
                            metas.all { kotlin.math.abs(it.appliedExpUs - expUs) <= 750L &&
                                it.appliedIso == grid.baseGain }
                        var checksum = 0L
                        for (frame in frames) {
                            var i = 0
                            while (i < frame.bayer.size) {
                                checksum = (checksum * 1_000_003L + frame.bayer[i]) and 0x7fff_ffffL
                                i += 4096
                            }
                        }
                        val profile = raw.lastCaptureProfile
                        val h = health.sample()
                        logger.row("exp511", listOf(
                            path, nBurst, rep, seqIdx, expUs,
                            if (error.isEmpty()) 1 else 0, "%.3f".format(wallMs),
                            profile?.let { fmt(it.firstResultMs) } ?: "",
                            profile?.let { fmt(it.firstImageMs) } ?: "",
                            profile?.let { fmt(it.firstMatchedMs) } ?: "",
                            profile?.let { fmt(it.decodeWallMs) } ?: "",
                            profile?.let { fmt(it.decodeCpuSumMs) } ?: "",
                            profile?.let { fmt(it.cleanupMs) } ?: "",
                            profile?.let { fmt(it.totalMs) } ?: "",
                            fmt(requestToApplied), fmt(sensorSpan), frames.size,
                            if (allMatch) 1 else 0,
                            if (path != "B_kotlin") raw.lastNativeDecodeMismatchCount else "",
                            checksum, error, h.batteryTemperatureC, h.thermalStatus, h.totalPssKb))
                        onStatus("EXP5.1.1 ${if (path.startsWith("B")) "B" else "D"} " +
                            "N=$nBurst ${rep * exposureSequence.size + seqIdx + 1}/" +
                            "${repeats * exposureSequence.size}")
                    }
                }
            }
        } finally {
            running = false
            logger.flush()
        }
        onStatus("EXP5.1.1 $path done")
    }

    /** EXP5.2 | Compare the optimized exact-N on-demand path with a bounded persistent
     * Camera2 listener/repeating stream and a six-frame metadata-confirmed RAW ring. */
    fun runExp52CaptureMode(
        onStatus: (String) -> Unit,
        path: String,
        repeats: Int = 5
    ) {
        require(path == "A_on_demand" || path == "B_continuous_ring")
        val exposureSequence = intArrayOf(32_000, 64_000, 128_000, 64_000, 32_000, 128_000)
        writeManifest("exp5_2_${path.lowercase()}", isGtReference = false,
            captureWidth = raw.captureWidth, captureHeight = raw.captureHeight,
            methodParams = JSONObject()
                .put("title", "EXP5.2 | on-demand vs bounded continuous RAW")
                .put("path", path)
                .put("camera_only", true)
                .put("ring_capacity", 6)
                .put("native_decode", true)
                .put("metadata_match_before_decode", true)
                .put("exposure_sequence_us", JSONArray(exposureSequence.toTypedArray()))
                .put("n_burst", JSONArray(arrayOf(1, 4)))
                .put("repeats", repeats))
        logger.csv("exp52", listOf(
            "path", "n_burst", "rep", "seq_idx", "exposure_req_us", "success",
            "wall_ms", "first_result_ms", "first_matched_ms", "ready_ms",
            "decode_cpu_sum_ms", "stale_frames_dropped", "decoded_frames",
            "returned_frames", "all_metadata_match", "sensor_span_ms",
            "sample_checksum", "error", "battery_temp_c", "thermal_status", "pss_kb"))
        fun fmt(value: Double): String = if (value.isNaN()) "" else "%.3f".format(value)
        running = true
        try {
            if (path == "B_continuous_ring") raw.startContinuousCapture()
            for (nBurst in intArrayOf(1, 4)) {
                // Unmeasured warm-up ensures both paths begin after request/listener setup.
                if (path == "B_continuous_ring")
                    raw.captureContinuous(48_000, grid.baseGain, nBurst)
                else raw.captureExp511Best(48_000, grid.baseGain, nBurst)
                for (rep in 0 until repeats) {
                    for ((seqIdx, expUs) in exposureSequence.withIndex()) {
                        if (!running) break
                        val wallStart = now()
                        var frames: List<RawFrame> = emptyList()
                        var error = ""
                        try {
                            frames = if (path == "B_continuous_ring")
                                raw.captureContinuous(expUs, grid.baseGain, nBurst)
                            else raw.captureExp511Best(expUs, grid.baseGain, nBurst)
                        } catch (t: Throwable) {
                            android.util.Log.e("EXP52", "$path N=$nBurst exp=$expUs failed", t)
                            error = (t.message ?: t.javaClass.simpleName)
                                .replace(',', ';').replace('\n', ' ')
                        }
                        val wallMs = ms(wallStart)
                        val metas = raw.lastMeta
                        val first = metas.firstOrNull(); val last = metas.lastOrNull()
                        val sensorSpan = if (first != null && last != null)
                            (last.timestamp - first.timestamp).coerceAtLeast(0L) / 1e6
                        else Double.NaN
                        val allMatch = frames.size == nBurst && metas.size == nBurst &&
                            metas.all { kotlin.math.abs(it.appliedExpUs - expUs) <= 750L &&
                                it.appliedIso == grid.baseGain }
                        var checksum = 0L
                        for (frame in frames) {
                            var i = 0
                            while (i < frame.bayer.size) {
                                checksum = (checksum * 1_000_003L + frame.bayer[i]) and 0x7fff_ffffL
                                i += 4096
                            }
                        }
                        val cp = raw.lastCaptureProfile
                        val sp = raw.lastContinuousProfile
                        val firstResult = if (path == "B_continuous_ring")
                            sp?.firstResultMs ?: Double.NaN else cp?.firstResultMs ?: Double.NaN
                        val firstMatched = if (path == "B_continuous_ring")
                            sp?.firstMatchedMs ?: Double.NaN else cp?.firstMatchedMs ?: Double.NaN
                        val ready = if (path == "B_continuous_ring")
                            sp?.readyMs ?: Double.NaN else cp?.totalMs ?: Double.NaN
                        val decodeCpu = if (path == "B_continuous_ring")
                            sp?.decodeCpuSumMs ?: Double.NaN else cp?.decodeCpuSumMs ?: Double.NaN
                        val h = health.sample()
                        logger.row("exp52", listOf(
                            path, nBurst, rep, seqIdx, expUs,
                            if (error.isEmpty()) 1 else 0, "%.3f".format(wallMs),
                            fmt(firstResult), fmt(firstMatched), fmt(ready), fmt(decodeCpu),
                            if (path == "B_continuous_ring") sp?.staleFramesDropped ?: 0 else 0,
                            if (path == "B_continuous_ring") sp?.decodedFrames ?: 0 else nBurst,
                            frames.size, if (allMatch) 1 else 0, fmt(sensorSpan), checksum, error,
                            h.batteryTemperatureC, h.thermalStatus, h.totalPssKb))
                        onStatus("EXP5.2 ${if (path.startsWith("A")) "A" else "B"} " +
                            "N=$nBurst ${rep * exposureSequence.size + seqIdx + 1}/" +
                            "${repeats * exposureSequence.size}")
                    }
                }
            }
        } finally {
            if (path == "B_continuous_ring") runCatching { raw.stopContinuousCapture() }
            running = false
            logger.flush()
        }
        onStatus("EXP5.2 $path done")
    }

    /** EXP5.3 | Compare every Camera2-advertised public RAW encoding at its advertised
     * resolution. RAW12 is logged as unsupported instead of being synthesized. */
    fun runExp53RawFormats(
        onStatus: (String) -> Unit,
        formatFilter: Int? = null,
        repeats: Int = 12
    ) {
        val configs = raw.availableRawStreamConfigs()
        writeManifest("exp5_3_raw_formats", isGtReference = false,
            captureWidth = raw.captureWidth, captureHeight = raw.captureHeight,
            methodParams = JSONObject()
                .put("title", "EXP5.3 | RAW format and resolution")
                .put("repeats_per_format_and_n", repeats)
                .put("capture_exposure_us", grid.fastestExposureUs)
                .put("capture_iso", grid.baseGain)
                .put("format_filter", formatFilter ?: "all")
                .put("native_decode", true)
                .put("pixel_sample_stride", 64))
        logger.csv("exp53_capabilities", listOf(
            "format", "format_id", "supported", "width", "height",
            "min_frame_duration_ns", "stall_duration_ns", "estimated_buffer_bytes"))
        val desired = listOf(
            ImageFormat.RAW_SENSOR to "RAW_SENSOR",
            ImageFormat.RAW10 to "RAW10",
            ImageFormat.RAW12 to "RAW12")
        for ((format, name) in desired) {
            val found = configs.filter { it.format == format }
            if (found.isEmpty()) logger.row("exp53_capabilities",
                listOf(name, format, 0, "", "", "", "", ""))
            else found.forEach { c -> logger.row("exp53_capabilities", listOf(
                c.formatName, c.format, 1, c.width, c.height,
                c.minFrameDurationNs, c.stallDurationNs, c.estimatedBufferBytes)) }
        }
        logger.csv("exp53", listOf(
            "format", "format_id", "width", "height", "n_burst", "rep", "success",
            "wall_ms", "sensor_to_image_ms", "first_result_ms", "decode_wall_ms",
            "decode_cpu_sum_ms", "actual_buffer_bytes", "estimated_buffer_bytes",
            "effective_decode_mb_s", "pss_kb", "formation_ms", "inference_ms",
            "candidate_count", "chosen_cell", "n_det", "sum_conf",
            "class_ids", "confidences", "boxes_xyxy",
            "pixel_mean", "pixel_std", "pixel_min", "pixel_max", "pixel_checksum",
            "paired_sample_mae", "paired_sample_rmse", "paired_sample_corr",
            "all_metadata_match", "native_mismatch_count", "error",
            "battery_temp_c", "thermal_status"))

        fun stats(values: IntArray): DoubleArray {
            var sum = 0.0; var sq = 0.0; var min = Int.MAX_VALUE; var max = Int.MIN_VALUE
            var checksum = 0L
            for (i in values.indices step 64) {
                val v = values[i]
                sum += v; sq += v.toDouble() * v
                if (v < min) min = v; if (v > max) max = v
                checksum = (checksum * 1_000_003L + v) and 0x7fff_ffffL
            }
            val count = (values.size + 63) / 64
            val mean = sum / count
            return doubleArrayOf(mean,
                kotlin.math.sqrt((sq / count - mean * mean).coerceAtLeast(0.0)),
                min.toDouble(), max.toDouble(), checksum.toDouble())
        }
        fun sample(values: IntArray): IntArray = IntArray((values.size + 63) / 64) {
            values[minOf(it * 64, values.lastIndex)]
        }
        fun compare(a: IntArray, b: IntArray): DoubleArray {
            val count = minOf(a.size, b.size)
            var abs = 0.0; var sq = 0.0; var sa = 0.0; var sb = 0.0
            for (i in 0 until count) {
                val d = a[i].toDouble() - b[i]
                abs += kotlin.math.abs(d); sq += d * d; sa += a[i]; sb += b[i]
            }
            val ma = sa / count; val mb = sb / count
            var cov = 0.0; var va = 0.0; var vb = 0.0
            for (i in 0 until count) {
                val da = a[i] - ma; val db = b[i] - mb
                cov += da * db; va += da * da; vb += db * db
            }
            val corr = if (va > 0.0 && vb > 0.0) cov / kotlin.math.sqrt(va * vb) else 0.0
            return doubleArrayOf(abs / count, kotlin.math.sqrt(sq / count), corr)
        }
        fun fmt(value: Double): String = if (value.isNaN()) "" else "%.4f".format(value)

        val rawSensorSamples = HashMap<Int, IntArray>()
        val supported = configs.filter {
            (it.format == ImageFormat.RAW_SENSOR || it.format == ImageFormat.RAW10) &&
                it.width == raw.captureWidth && it.height == raw.captureHeight &&
                (formatFilter == null || it.format == formatFilter)
        }.sortedBy { if (it.format == ImageFormat.RAW_SENSOR) 0 else 1 }
        val source = ParallelRawCandidateSource(grid, raw,
            colorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB,
            burstWindow = BurstWindow.FIRST_N)
        running = true
        try {
            for (config in supported) {
                if (!running) break
                onStatus("EXP5.3 configure ${config.formatName}")
                raw.reconfigureRawStream(config.format, config.width, config.height)
                for (nBurst in intArrayOf(1, 4)) {
                    raw.armNativeDecodeValidation()
                    raw.captureNoGuardNative(grid.fastestExposureUs, grid.baseGain, nBurst)
                    check(raw.lastNativeDecodeMismatchCount == 0) {
                        "${config.formatName} native decode mismatch=" +
                            raw.lastNativeDecodeMismatchCount
                    }
                    for (rep in 0 until repeats) {
                        if (!running) break
                        val wallStart = now()
                        var frames: List<RawFrame> = emptyList()
                        var error = ""
                        try {
                            frames = raw.captureNoGuardNative(
                                grid.fastestExposureUs, grid.baseGain, nBurst)
                        } catch (t: Throwable) {
                            error = (t.message ?: t.javaClass.simpleName)
                                .replace(',', ';').replace('\n', ' ')
                        }
                        val wallMs = ms(wallStart)
                        var formationMs = Double.NaN; var inferenceMs = Double.NaN
                        var detections: List<Detection> = emptyList()
                        var chosenCell = -1
                        var candidateCount = 0
                        if (frames.isNotEmpty()) {
                            val cells = if (nBurst == 1)
                                IntArray(grid.nGain) { grid.cell(it, 0) }
                            else IntArray(grid.nGain * grid.nShutter) { it }
                            candidateCount = cells.size
                            val tf = now()
                            val images = source.formAllCells(frames, cells)
                            formationMs = ms(tf)
                            val ti = now(); val byCell = detector.detectBatch(images)
                            inferenceMs = ms(ti)
                            val scores = DoubleArray(byCell.size) { i ->
                                byCell[i].sumOf {
                                    if (it.confidence >= selectConf) it.confidence.toDouble() else 0.0
                                }
                            }
                            var chosen = 0
                            for (i in 1 until scores.size) if (scores[i] > scores[chosen]) chosen = i
                            chosenCell = cells[chosen]
                            detections = byCell[chosen]
                            images.forEach { it.recycle() }
                        }
                        val frame = frames.lastOrNull()
                        val pixelStats = frame?.let { stats(it.bayer) }
                            ?: doubleArrayOf(Double.NaN, Double.NaN, Double.NaN,
                                Double.NaN, Double.NaN)
                        val key = nBurst * 1000 + rep
                        var paired = doubleArrayOf(Double.NaN, Double.NaN, Double.NaN)
                        if (frame != null) {
                            val sampled = sample(frame.bayer)
                            if (config.format == ImageFormat.RAW_SENSOR) {
                                rawSensorSamples[key] = sampled
                            } else {
                                rawSensorSamples[key]?.let { paired = compare(it, sampled) }
                            }
                        }
                        val profile = raw.lastCaptureProfile
                        val meta = raw.lastMeta.lastOrNull()
                        val actualBytes = meta?.rowStrideBytes?.toLong()?.times(config.height) ?: 0L
                        val bandwidth = if (profile != null && profile.decodeCpuSumMs > 0.0)
                            actualBytes / 1_048_576.0 / (profile.decodeCpuSumMs / 1000.0)
                        else Double.NaN
                        val allMatch = frames.size == nBurst && raw.lastMeta.size == nBurst &&
                            raw.lastMeta.all {
                                kotlin.math.abs(it.appliedExpUs - grid.fastestExposureUs) <= 750L &&
                                    it.appliedIso == grid.baseGain }
                        val kept = detections.filter { it.confidence >= selectConf }
                        val h = health.sample()
                        logger.row("exp53", listOf(
                            config.formatName, config.format, config.width, config.height,
                            nBurst, rep, if (error.isEmpty()) 1 else 0,
                            "%.3f".format(wallMs), profile?.let { fmt(it.firstImageMs) } ?: "",
                            profile?.let { fmt(it.firstResultMs) } ?: "",
                            profile?.let { fmt(it.decodeWallMs) } ?: "",
                            profile?.let { fmt(it.decodeCpuSumMs) } ?: "",
                            actualBytes, config.estimatedBufferBytes, fmt(bandwidth), h.totalPssKb,
                            fmt(formationMs), fmt(inferenceMs), candidateCount, chosenCell, kept.size,
                            fmt(kept.sumOf { it.confidence.toDouble() }),
                            kept.joinToString(";") { it.classId.toString() },
                            kept.joinToString(";") { "%.4f".format(it.confidence) },
                            kept.joinToString("|") { d -> d.xyxy.joinToString(";") { "%.1f".format(it) } },
                            fmt(pixelStats[0]), fmt(pixelStats[1]), pixelStats[2].toInt(),
                            pixelStats[3].toInt(), pixelStats[4].toLong(),
                            fmt(paired[0]), fmt(paired[1]), fmt(paired[2]),
                            if (allMatch) 1 else 0, raw.lastNativeDecodeMismatchCount, error,
                            h.batteryTemperatureC, h.thermalStatus))
                        onStatus("EXP5.3 ${config.formatName} N=$nBurst ${rep + 1}/$repeats")
                    }
                }
            }
        } finally {
            runCatching {
                val sensor = configs.firstOrNull { it.format == ImageFormat.RAW_SENSOR }
                if (sensor != null && raw.streamFormat != ImageFormat.RAW_SENSOR)
                    raw.reconfigureRawStream(sensor.format, sensor.width, sensor.height)
            }
            source.shutdown(); running = false; logger.flush()
        }
        onStatus("EXP5.3 done")
    }

    /** EXP5.4 | CPU contention and optional-work ablation on the best EXP5 capture path.
     * Stage 1 isolates Camera2 + RAW10 native decode. Stage 2 overlaps the next exact-N
     * capture with current K=3/K=9 formation, optional GPU, display bitmap, overlay and I/O. */
    fun runExp54CpuContention(
        onStatus: (String) -> Unit,
        confirmationOnly: Boolean = false,
        cameraN1Repeats: Int = 12,
        cameraN4Repeats: Int = 8,
        measuredCycles: Int = 3
    ) {
        data class Arm(
            val name: String,
            val decodeThreads: Int = 4,
            val formationThreads: Int = 4,
            val gpu: Boolean = true,
            val display: Boolean = false,
            val overlay: Boolean = false,
            val storage: Boolean = false
        )
        data class Packet(
            val startedNs: Long,
            val endedNs: Long,
            val frames: List<RawFrame>,
            val profile: RawSensorCapturer.CaptureProfile?
        )

        val baseArm = Arm("BASE_d4_f4_gpu_on_display_off")
        val bestArm = Arm("BEST_d2_f3_gpu_on_display_off",
            decodeThreads = 2, formationThreads = 3)
        val arms = if (confirmationOnly) listOf(baseArm, bestArm, bestArm, baseArm) else listOf(
            baseArm,
            Arm("DISPLAY_on", display = true),
            Arm("OVERLAY_on", display = true, overlay = true),
            Arm("STORAGE_raw_jpeg_on", display = true, storage = true),
            Arm("DECODE_t1", decodeThreads = 1),
            Arm("DECODE_t2", decodeThreads = 2),
            Arm("FORMATION_t2", formationThreads = 2),
            Arm("FORMATION_t3", formationThreads = 3),
            Arm("GPU_off", gpu = false))
        val configs = raw.availableRawStreamConfigs()
        val raw10 = configs.firstOrNull { it.format == ImageFormat.RAW10 }
            ?: error("EXP5.4 requires RAW10")
        val rawSensor = configs.firstOrNull { it.format == ImageFormat.RAW_SENSOR }
        writeManifest("exp5_4_cpu_contention", isGtReference = false,
            captureWidth = raw10.width, captureHeight = raw10.height,
            methodParams = JSONObject()
                .put("title", "EXP5.4 | CPU contention and optional work")
                .put("raw_format", "RAW10")
                .put("capture", "exact-N native on-demand")
                .put("camera_n1_repeats", cameraN1Repeats)
                .put("camera_n4_repeats", cameraN4Repeats)
                .put("measured_p5_cycles", measuredCycles)
                .put("warmup_p5_cycles", 1)
                .put("confirmation_only", confirmationOnly)
                .put("arms", JSONArray(arms.map { it.name })))
        logger.csv("exp54_camera", listOf(
            "decode_threads", "n_burst", "rep", "wall_ms", "first_result_ms",
            "sensor_to_image_ms", "decode_wall_ms", "decode_cpu_sum_ms",
            "actual_buffer_bytes", "pss_kb", "battery_temp_c", "thermal_status",
            "success", "all_metadata_match", "error"))
        logger.csv("exp54_overlap", listOf(
            "arm", "decode_threads", "formation_threads", "gpu_on", "display_on",
            "overlay_on", "storage_on", "step", "cycle", "phase", "k", "n_burst",
            "capture_wall_ms", "capture_wait_ms", "capture_hidden_ms",
            "first_image_ms", "decode_cpu_sum_ms", "formation_ms", "gpu_ms",
            "yolo_decode_ms", "inference_ms", "detection_ready_ms",
            "selected_bitmap_ms", "overlay_ms", "storage_ms", "post_detection_ms",
            "step_result_ms", "n_det", "sum_conf", "chosen_cell",
            "pss_kb", "battery_temp_c", "thermal_status"))
        fun fmt(value: Double): String = if (value.isNaN()) "" else "%.3f".format(value)
        val yolo = detector as? com.example.activeperception.acquire.TfliteYoloDetector
            ?: error("EXP5.4 requires TfliteYoloDetector")
        running = true
        try {
            raw.reconfigureRawStream(raw10.format, raw10.width, raw10.height)
            raw.armNativeDecodeValidation()
            raw.captureNoGuardNative(grid.fastestExposureUs, grid.baseGain, 1)
            check(raw.lastNativeDecodeMismatchCount == 0)

            // Stage 1: camera-only. No formation, GPU, Bitmap, overlay, or file output.
            if (!confirmationOnly) {
                for (threads in intArrayOf(1, 2, 4)) {
                    raw.configureDecodeThreads(threads)
                    for ((nBurst, repeats) in listOf(1 to cameraN1Repeats, 4 to cameraN4Repeats)) {
                        raw.captureNoGuardNative(grid.fastestExposureUs, grid.baseGain, nBurst)
                        for (rep in 0 until repeats) {
                        if (!running) break
                        val started = now(); var error = ""; var frames: List<RawFrame> = emptyList()
                        try {
                            frames = raw.captureNoGuardNative(
                                grid.fastestExposureUs, grid.baseGain, nBurst)
                        } catch (t: Throwable) {
                            error = (t.message ?: t.javaClass.simpleName)
                                .replace(',', ';').replace('\n', ' ')
                        }
                        val wall = ms(started); val profile = raw.lastCaptureProfile
                        val metas = raw.lastMeta
                        val allMatch = frames.size == nBurst && metas.size == nBurst && metas.all {
                            kotlin.math.abs(it.appliedExpUs - grid.fastestExposureUs) <= 750L &&
                                it.appliedIso == grid.baseGain }
                        val actualBytes = metas.firstOrNull()?.rowStrideBytes?.toLong()
                            ?.times(raw10.height) ?: 0L
                        val h = health.sample()
                        logger.row("exp54_camera", listOf(
                            threads, nBurst, rep, fmt(wall),
                            profile?.let { fmt(it.firstResultMs) } ?: "",
                            profile?.let { fmt(it.firstImageMs) } ?: "",
                            profile?.let { fmt(it.decodeWallMs) } ?: "",
                            profile?.let { fmt(it.decodeCpuSumMs) } ?: "",
                            actualBytes, h.totalPssKb, h.batteryTemperatureC, h.thermalStatus,
                            if (error.isEmpty()) 1 else 0, if (allMatch) 1 else 0, error))
                        onStatus("EXP5.4 camera d$threads N=$nBurst ${rep + 1}/$repeats")
                        }
                    }
                }
            }

            raw.configureDecodeThreads(4)
            yolo.warmUpAllBatches()
            val totalSteps = (measuredCycles + 1) * 5
            for ((armIndex, arm) in arms.withIndex()) {
                if (!running) break
                raw.configureDecodeThreads(arm.decodeThreads)
                val source = ParallelRawCandidateSource(grid, raw,
                    nThreads = arm.formationThreads,
                    colorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB,
                    burstWindow = BurstWindow.FIRST_N)
                val cameraExecutor = Executors.newSingleThreadExecutor()
                fun submit(nBurst: Int): Future<Packet> = cameraExecutor.submit<Packet> {
                    val start = now()
                    val frames = raw.captureNoGuardNative(
                        grid.fastestExposureUs, grid.baseGain, nBurst)
                    Packet(start, now(), frames, raw.lastCaptureProfile)
                }
                try {
                    var current = submit(4)
                    for (step in 0 until totalSteps) {
                        if (!running) break
                        val stepStart = now()
                        val phaseIndex = step % 5
                        val isProbe = phaseIndex == 0
                        val nBurst = if (isProbe) 4 else 1
                        val cells = if (isProbe)
                            IntArray(grid.nGain * grid.nShutter) { it }
                        else IntArray(grid.nGain) { grid.cell(it, 0) }
                        val waitStart = now(); val packet = current.get(); val captureWait = ms(waitStart)
                        val captureWall = (packet.endedNs - packet.startedNs) / 1e6
                        val captureHidden = (captureWall - captureWait).coerceAtLeast(0.0)

                        // Queue next RAW before any formation/GPU/display/I/O work.
                        val nextN = if ((step + 1) % 5 == 0) 4 else 1
                        val next = if (step + 1 < totalSteps) submit(nextN) else null
                        val formationStart = now()
                        val prepared = source.prepareExp21Rgb(packet.frames, cells)
                        val tensor = source.formExp21NativeTensor(prepared)
                        val formationMs = ms(formationStart)
                        var detections: List<List<Detection>> = List(cells.size) { emptyList() }
                        var gpuMs = 0.0; var yoloDecodeMs = 0.0; var inferenceMs = 0.0
                        if (arm.gpu) {
                            detections = yolo.detectTensorBatchOptimized(tensor.batch)
                            gpuMs = yolo.lastRunMs; yoloDecodeMs = yolo.lastDecodeMs
                            inferenceMs = yolo.lastPreprocessMs + gpuMs + yoloDecodeMs
                        }
                        val scores = DoubleArray(cells.size) { lane ->
                            detections[lane].sumOf {
                                if (it.confidence >= selectConf) it.confidence.toDouble() else 0.0
                            }
                        }
                        var chosen = 0
                        for (i in 1 until scores.size) if (scores[i] > scores[chosen]) chosen = i
                        val detectionReadyNs = now()

                        var selectedBitmap: Bitmap? = null
                        var bitmapMs = 0.0; var overlayMs = 0.0; var storageMs = 0.0
                        if (arm.display || arm.overlay || arm.storage) {
                            val tb = now()
                            selectedBitmap = source.formExp21Bitmaps(prepared.copy(
                                cells = intArrayOf(cells[chosen]))).images[0]
                            bitmapMs = ms(tb)
                        }
                        if (arm.overlay) {
                            val to = now()
                            val overlay = drawDetectionOverlay(requireNotNull(selectedBitmap),
                                detections[chosen].filter { it.confidence >= selectConf })
                            overlay.recycle(); overlayMs = ms(to)
                        }
                        if (arm.storage) {
                            val ts = now()
                            val suffix = "a${armIndex}_s${step}"
                            logger.saveRaw("exp54_$suffix", packet.frames.last().bayer)
                            logger.saveJpeg("exp54_$suffix", requireNotNull(selectedBitmap), 90)
                            storageMs = ms(ts)
                        }
                        selectedBitmap?.recycle()
                        val resultNs = now()
                        if (step >= 5) {
                            val selected = detections[chosen].filter { it.confidence >= selectConf }
                            val h = health.sample()
                            logger.row("exp54_overlap", listOf(
                                arm.name, arm.decodeThreads, arm.formationThreads,
                                if (arm.gpu) 1 else 0, if (arm.display) 1 else 0,
                                if (arm.overlay) 1 else 0, if (arm.storage) 1 else 0,
                                step - 5, (step - 5) / 5,
                                if (isProbe) "burst" else "single", cells.size, nBurst,
                                fmt(captureWall), fmt(captureWait), fmt(captureHidden),
                                packet.profile?.let { fmt(it.firstImageMs) } ?: "",
                                packet.profile?.let { fmt(it.decodeCpuSumMs) } ?: "",
                                fmt(formationMs), fmt(gpuMs), fmt(yoloDecodeMs), fmt(inferenceMs),
                                fmt((detectionReadyNs - stepStart) / 1e6), fmt(bitmapMs),
                                fmt(overlayMs), fmt(storageMs),
                                fmt((resultNs - detectionReadyNs) / 1e6),
                                fmt((resultNs - stepStart) / 1e6), selected.size,
                                fmt(selected.sumOf { it.confidence.toDouble() }), cells[chosen],
                                h.totalPssKb, h.batteryTemperatureC, h.thermalStatus))
                        }
                        if (next != null) current = next
                        onStatus("EXP5.4 ${arm.name} ${step + 1}/$totalSteps")
                    }
                } finally {
                    cameraExecutor.shutdownNow(); source.shutdown()
                }
            }
        } finally {
            runCatching { raw.configureDecodeThreads(4) }
            runCatching {
                if (rawSensor != null && raw.streamFormat != ImageFormat.RAW_SENSOR)
                    raw.reconfigureRawStream(rawSensor.format, rawSensor.width, rawSensor.height)
            }
            running = false; logger.flush()
        }
        onStatus("EXP5.4 done")
    }

    /** Production entry point. It keeps the paper's P-period policy while selecting the
     * experimentally validated RayNeo fast path for every K=9/K=3 step. */
    fun runFinalProposed(
        period: Int,
        maxFrames: Int,
        fallback: FallbackMetric,
        onStatus: (String) -> Unit,
        onFrame: (Bitmap, List<Detection>, Int, Int) -> Unit
    ) = runExp55FinalAdaptiveP5(
        onStatus = onStatus,
        warmupCycles = 0,
        measuredCycles = (maxFrames + period - 1) / period,
        persistentFastCapture = true,
        decodeThreads = 4,
        formationThreads = 4,
        deepSinglePrefetch = true,
        integratedOptimizations = true,
        productionMode = true,
        period = period,
        maxFrames = maxFrames,
        fallback = fallback,
        onFrame = onFrame
    )

    /** EXP5.5 | Final adaptive validation after EXP5. The probe chooses a shutter row,
     * requests that physical exposure immediately, then runs four single-frame K=3 steps.
     * Display, overlay and image persistence stay outside detection-ready. */
    fun runExp55FinalAdaptiveP5(
        onStatus: (String) -> Unit,
        warmupCycles: Int = 2,
        measuredCycles: Int = 10,
        persistentFastCapture: Boolean = false,
        decodeThreads: Int = 4,
        formationThreads: Int = 4,
        deepSinglePrefetch: Boolean = false,
        integratedOptimizations: Boolean = false,
        productionMode: Boolean = false,
        period: Int = 5,
        maxFrames: Int = (warmupCycles + measuredCycles) * period,
        fallback: FallbackMetric = FallbackMetric.SAFE_CELL,
        onFrame: (Bitmap, List<Detection>, Int, Int) -> Unit = { _, _, _, _ -> }
    ) {
        require(period >= 2)
        data class Packet(
            val requestedExpUs: Int,
            val nBurst: Int,
            val startedNs: Long,
            val startedSensorNs: Long,
            val endedNs: Long,
            val frames: List<RawFrame>,
            val metas: List<RawSensorCapturer.RawMeta>,
            val profile: RawSensorCapturer.CaptureProfile?
        )
        val configs = raw.availableRawStreamConfigs()
        val raw10 = configs.firstOrNull { it.format == ImageFormat.RAW10 }
            ?: error("EXP5.5 requires RAW10")
        val rawSensor = configs.firstOrNull { it.format == ImageFormat.RAW_SENSOR }
        val optimizedDetector = detector as? com.example.activeperception.acquire.TfliteYoloDetector
            ?: error("EXP5.5 requires TfliteYoloDetector")
        val fastEnabled = persistentFastCapture || deepSinglePrefetch
        writeManifest(if (productionMode) "rayneo_sos_final"
            else if (integratedOptimizations) "exp6_integrated_compute"
            else if (deepSinglePrefetch) "exp5_7_streaming_single_prefetch"
            else if (persistentFastCapture) "exp5_6_fast_capture_p5"
            else "exp5_5_final_adaptive_p5", isGtReference = false,
            captureWidth = raw10.width, captureHeight = raw10.height,
            methodParams = JSONObject()
                .put("title", if (productionMode)
                    "RayNeo SoS Final | optimized adaptive pipeline"
                    else if (integratedOptimizations)
                    "EXP6 | fused formation + async health + best available FP16 GPU"
                    else if (deepSinglePrefetch)
                    "EXP5.7 | four-single finite streaming prefetch"
                    else if (persistentFastCapture)
                    "EXP5.6 | persistent finite callback adaptive P=5"
                    else "EXP5.5 | final RAW10 adaptive P=5")
                .put("warmup_cycles", warmupCycles)
                .put("measured_cycles", measuredCycles)
                .put("period", period)
                .put("max_frames", maxFrames)
                .put("fallback_metric", fallback.tieBreakName())
                .put("capture", if (deepSinglePrefetch)
                    "N4 burst probe; argmax; one finite four-frame single sequence streamed into formation"
                    else if (persistentFastCapture)
                    "persistent finite exact-N; dedicated result/image callbacks; preview N1; still N4"
                    else "exact-N on-demand; preview N1; still N4")
                .put("decode_threads", decodeThreads)
                .put("formation_threads", formationThreads)
                .put("deep_single_prefetch", deepSinglePrefetch)
                .put("formation_path", if (integratedOptimizations)
                    "fused Bayer sum+demosaic+gain+sRGB+letterbox native"
                    else "shared RGB then native tensor")
                .put("health_sampling", if (integratedOptimizations)
                    "2s low-priority background cache" else "cached")
                .put("yolo_choice", "640 FP16 GPU COCO5 B3; K9 executes as 3xB3 to avoid clean-cache B9 delegate hang")
                .put("gpu_model", "COCO5 FP16 B1/B3")
                .put("display_overlay_image_save_in_critical_path", false)
                .put("physical_exposure_adaptation", true))
        logger.csv("exp55", listOf(
            "measured", "cycle", "step", "phase", "k", "n_burst",
            "requested_exp_us", "applied_exp_us", "applied_iso", "metadata_match",
            "apply_delay_frames", "request_to_applied_ms", "capture_wall_ms",
            "capture_wait_ms", "capture_hidden_ms", "first_image_ms",
            "decode_wall_ms", "decode_cpu_sum_ms", "formation_ms", "gpu_ms",
            "yolo_decode_ms", "inference_ms", "detection_ready_step_ms",
            "result_interval_ms", "cycle_interval_ms", "cycle_fps",
            "chosen_cell", "chosen_shutter_idx", "next_exp_us", "n_det", "sum_conf",
            "pss_kb", "battery_temp_c", "thermal_status"))
        logger.csv("exp6_validation", listOf(
            "samples", "mismatch_gt_1e5", "mean_abs", "max_abs"))
        if (productionMode) {
            logger.csv("candidates", listOf("frame", "cell", "gain", "exposure_us",
                "sum_conf", "chosen", "tie_break"))
            logger.csv("router", listOf("frame", "score", "model_limited_clusters",
                "recovered_locally_clusters", "offload"))
        }
        fun fmt(v: Double): String = if (v.isNaN()) "" else "%.3f".format(v)
        val source = ParallelRawCandidateSource(grid, raw, nThreads = formationThreads,
            colorPipeline = ColorPipeline.ORIGINAL_GAIN_SRGB,
            burstWindow = BurstWindow.FIRST_N)
        val cameraExecutor = Executors.newSingleThreadExecutor()
        val router = CrossExposureRouter(selectConf)
        running = true
        try {
            raw.reconfigureRawStream(raw10.format, raw10.width, raw10.height)
            raw.configureDecodeThreads(decodeThreads)
            if (fastEnabled) raw.startFastCapture()
            raw.armNativeDecodeValidation()
            if (fastEnabled)
                raw.captureFast(grid.fastestExposureUs, grid.baseGain, 1)
            else raw.captureExp511Best(grid.fastestExposureUs, grid.baseGain, 1)
            check(raw.lastNativeDecodeMismatchCount == 0)
            onStatus("RayNeo SoS · warming FP16 GPU B=1/3")
            optimizedDetector.warmUpAllBatches()

            fun submit(exposureUs: Int, nBurst: Int): Future<Packet> =
                cameraExecutor.submit<Packet> {
                    val start = now(); val sensorStart = SystemClock.elapsedRealtimeNanos()
                    val frames = if (fastEnabled)
                        raw.captureFast(exposureUs, grid.baseGain, nBurst)
                    else raw.captureExp511Best(exposureUs, grid.baseGain, nBurst)
                    val snapshot = if (fastEnabled) raw.consumeCaptureSnapshot() else null
                    Packet(exposureUs, nBurst, start, sensorStart, now(), frames,
                        (snapshot?.metas ?: raw.lastMeta).map { it.copy(
                            blackLevels = it.blackLevels.copyOf(),
                            whiteBalance = it.whiteBalance.copyOf(),
                            cameraToSrgb = it.cameraToSrgb.copyOf()) },
                        snapshot?.profile ?: raw.lastCaptureProfile)
                }

            val totalCycles = (maxFrames + period - 1) / period
            val totalSteps = maxFrames
            var anchor = grid.cell(grid.nGain - 1, 0)
            var current = submit(grid.fastestExposureUs, 4)
            var singleSequence: RawSensorCapturer.FastSequenceHandle? = null
            var previousResultNs = -1L
            var previousCycleEndNs = -1L
            var fusedValidated = false
            for (globalStep in 0 until totalSteps) {
                if (!running) break
                val cycle = globalStep / period
                val phaseIndex = globalStep % period
                val isProbe = phaseIndex == 0
                val measured = cycle >= warmupCycles
                val stepStart = now()
                val cells = if (isProbe)
                    IntArray(grid.nGain * grid.nShutter) { it }
                else {
                    val sj = grid.indices(anchor).second
                    IntArray(grid.nGain) { gi -> grid.cell(gi, sj) }
                }
                val waitStart = now()
                val packet = if (!isProbe && deepSinglePrefetch) {
                    val handle = requireNotNull(singleSequence)
                    val frame = raw.takeFastSequenceFrame(handle)
                    Packet(handle.requestedExpUs, 1, handle.startedNs,
                        handle.startedSensorNs, now(), listOf(frame),
                        raw.lastMeta.map { it.copy(
                            blackLevels = it.blackLevels.copyOf(),
                            whiteBalance = it.whiteBalance.copyOf(),
                            cameraToSrgb = it.cameraToSrgb.copyOf()) },
                        raw.lastCaptureProfile)
                } else current.get()
                val captureWait = ms(waitStart)
                val captureWall = (packet.endedNs - packet.startedNs) / 1e6
                val captureHidden = (captureWall - captureWait).coerceAtLeast(0.0)
                val meta = packet.metas.firstOrNull()
                val metadataMatch = packet.metas.size == packet.nBurst && packet.metas.all {
                    kotlin.math.abs(it.appliedExpUs - packet.requestedExpUs) <= 750L &&
                        it.appliedIso == grid.baseGain }
                val requestToApplied = meta?.timestamp?.let {
                    (it - packet.startedSensorNs).coerceAtLeast(0L) / 1e6
                } ?: Double.NaN

                // For singles the next state is already known. Queue it before processing.
                var next: Future<Packet>? = null
                if (!isProbe && globalStep + 1 < totalSteps &&
                    !deepSinglePrefetch) {
                    val nextIsProbe = (phaseIndex == period - 1)
                    val nextExp = if (nextIsProbe) grid.fastestExposureUs
                        else grid.exposuresUs[grid.indices(anchor).second]
                    next = submit(nextExp, if (nextIsProbe) 4 else 1)
                }

                val formationStart = now()
                val tensor = if (integratedOptimizations) {
                    if (!productionMode && !fusedValidated && isProbe) {
                        val reference = source.formExp21NativeTensor(
                            source.prepareExp21Rgb(packet.frames, cells))
                        val totalFloats = cells.size * 640 * 640 * 3
                        val sampleCount = minOf(20_000, totalFloats)
                        val indices = IntArray(sampleCount) { i ->
                            if (sampleCount == 1) 0 else
                                ((i.toLong() * (totalFloats - 1)) / (sampleCount - 1)).toInt()
                        }
                        val values = FloatArray(sampleCount) { i ->
                            reference.batch.input.getFloat(indices[i] * 4)
                        }
                        val fused = source.formFusedNativeTensor(packet.frames, cells)
                        var mismatch = 0; var sumAbs = 0.0; var maxAbs = 0.0
                        for (i in indices.indices) {
                            val d = kotlin.math.abs(
                                fused.batch.input.getFloat(indices[i] * 4) - values[i])
                            if (d > 1e-5) mismatch++
                            sumAbs += d; if (d > maxAbs) maxAbs = d.toDouble()
                        }
                        logger.row("exp6_validation", listOf(sampleCount, mismatch,
                            fmt(sumAbs / sampleCount), fmt(maxAbs)))
                        check(mismatch == 0) {
                            "EXP6 fused tensor mismatch=$mismatch/$sampleCount max=$maxAbs"
                        }
                        fusedValidated = true
                        fused
                    } else source.formFusedNativeTensor(packet.frames, cells)
                } else {
                    val prepared = source.prepareExp21Rgb(packet.frames, cells)
                    source.formExp21NativeTensor(prepared)
                }
                val formationMs = ms(formationStart)
                val active = optimizedDetector
                val detections = active.detectTensorBatchOptimizedFlexible(tensor.batch)
                val scores = DoubleArray(cells.size) { lane ->
                    detections[lane].sumOf {
                        if (it.confidence >= selectConf) it.confidence.toDouble() else 0.0
                    }
                }
                var chosen = 0
                for (i in 1 until scores.size) if (scores[i] > scores[chosen]) chosen = i
                var tieBreak = "conf"
                var candidateBitmaps: List<Bitmap>? = null
                if (scores[chosen] == 0.0) {
                    tieBreak = fallback.tieBreakName()
                    candidateBitmaps = cells.indices.map { source.selectedTensorBitmap(tensor, it) }
                    if (fallback == FallbackMetric.SAFE_CELL) {
                        var pick = -1; var bestDist = Double.MAX_VALUE
                        for (i in cells.indices) {
                            if (grid.indices(cells[i]).second != 0) continue
                            val mean = meanLumaRatio(candidateBitmaps[i])
                            val distance = kotlin.math.abs(
                                kotlin.math.ln((mean + 1e-6) / SAFE_TARGET_RATIO))
                            if (distance < bestDist) { bestDist = distance; pick = i }
                        }
                        if (pick < 0) pick = cells.indices.maxByOrNull {
                            grid.indices(cells[it]).first
                        } ?: 0
                        chosen = pick
                    } else {
                        val fallbackValues = fallbackScores(requireNotNull(candidateBitmaps), fallback)
                        chosen = fallbackValues.indices.maxByOrNull { fallbackValues[it] } ?: 0
                    }
                }
                anchor = cells[chosen]
                val chosenSj = grid.indices(anchor).second
                val selected = detections[chosen].filter { it.confidence >= selectConf }
                val detectionReadyNs = now()

                // Probe→single is data-dependent and is queued at the first point after argmax.
                if (isProbe && globalStep + 1 < totalSteps) {
                    if (deepSinglePrefetch) {
                        val singles = minOf(period - 1, totalSteps - globalStep - 1)
                        singleSequence = raw.beginFastSequence(
                            grid.exposuresUs[chosenSj], grid.baseGain, singles)
                        // The four finite single requests are already ordered in Camera2.
                        // Queue the next known fastest-exposure burst directly behind them so
                        // all four single formation/inference steps hide its sensor delivery.
                        if (globalStep + singles + 1 < totalSteps)
                            next = submit(grid.fastestExposureUs, 4)
                    } else next = submit(grid.exposuresUs[chosenSj], 1)
                }
                val resultInterval = if (previousResultNs < 0L) Double.NaN
                    else (detectionReadyNs - previousResultNs) / 1e6
                val cycleInterval = if (phaseIndex == period - 1 && previousCycleEndNs > 0L)
                    (detectionReadyNs - previousCycleEndNs) / 1e6 else Double.NaN
                val cycleFps = if (cycleInterval.isFinite() && cycleInterval > 0.0)
                    period * 1_000.0 / cycleInterval else Double.NaN
                if (phaseIndex == period - 1) previousCycleEndNs = detectionReadyNs
                previousResultNs = detectionReadyNs
                val nextExp = if (phaseIndex == period - 1) grid.fastestExposureUs
                    else grid.exposuresUs[chosenSj]
                val h = health.sample()
                logger.row("exp55", listOf(
                    if (measured) 1 else 0, cycle - warmupCycles, phaseIndex,
                    if (isProbe) "burst_probe" else "single", cells.size, packet.nBurst,
                    packet.requestedExpUs, meta?.appliedExpUs ?: -1,
                    meta?.appliedIso ?: -1, if (metadataMatch) 1 else 0,
                    meta?.applyDelayFrames ?: -1, fmt(requestToApplied), fmt(captureWall),
                    fmt(captureWait), fmt(captureHidden),
                    packet.profile?.let { fmt(it.firstImageMs) } ?: "",
                    packet.profile?.let { fmt(it.decodeWallMs) } ?: "",
                    packet.profile?.let { fmt(it.decodeCpuSumMs) } ?: "",
                    fmt(formationMs), fmt(active.lastRunMs), fmt(active.lastDecodeMs),
                    fmt(active.lastPreprocessMs + active.lastRunMs + active.lastDecodeMs),
                    fmt((detectionReadyNs - stepStart) / 1e6), fmt(resultInterval),
                    fmt(cycleInterval), fmt(cycleFps), anchor, chosenSj, nextExp,
                    selected.size, fmt(selected.sumOf { it.confidence.toDouble() }),
                    h.totalPssKb, h.batteryTemperatureC, h.thermalStatus))
                if (productionMode) {
                    lastRoutingDecision = router.decide(detections)
                    logger.row("router", listOf(globalStep,
                        "%.5f".format(lastRoutingDecision.score),
                        lastRoutingDecision.modelLimitedClusters,
                        lastRoutingDecision.recoveredLocallyClusters,
                        if (lastRoutingDecision.shouldOffload) 1 else 0))
                    for (i in cells.indices) {
                        val (gi, sj) = grid.indices(cells[i])
                        logger.row("candidates", listOf(globalStep, cells[i],
                            effIso(grid.gains[gi]), grid.exposuresUs[sj],
                            "%.5f".format(scores[i]), if (i == chosen) 1 else 0,
                            if (i == chosen) tieBreak else ""))
                        val detArray = JSONArray()
                        for (d in detections[i]) detArray.put(detJson(d))
                        logger.jsonl("candidate_dets", JSONObject().apply {
                            put("frame", globalStep); put("cell", cells[i])
                            put("chosen", i == chosen); put("sum_conf", scores[i])
                            put("dets", detArray)
                        })
                    }
                    val bitmap = candidateBitmaps?.get(chosen)
                        ?: source.selectedTensorBitmap(tensor, chosen)
                    val previewDets = source.detectionsForTensorPreview(selected, tensor, chosen)
                    val imagePath = logger.saveJpegAsync(
                        "${frameName(globalStep)}_cell${anchor}", bitmap)
                    logFrame(globalStep, "proposed_final", anchor,
                        grid.gains[grid.indices(anchor).first], grid.exposuresUs[chosenSj],
                        cells.size, isProbe, formationMs,
                        active.lastPreprocessMs + active.lastRunMs + active.lastDecodeMs,
                        (detectionReadyNs - stepStart) / 1e6,
                        previewDets, imagePath, tieBreak)
                    onFrame(bitmap, previewDets, effIso(grid.gains[grid.indices(anchor).first]),
                        grid.exposuresUs[chosenSj])
                }
                if (globalStep + 1 < totalSteps && next != null) current = next
                onStatus("RayNeo SoS · ${globalStep + 1}/$totalSteps · " +
                    "${if (isProbe) "K=9 burst" else "K=3 single"} · cell=$anchor")
            }
        } finally {
            cameraExecutor.shutdownNow(); source.shutdown()
            runCatching { raw.stopFastCapture() }
            runCatching {
                if (rawSensor != null && raw.streamFormat != ImageFormat.RAW_SENSOR)
                    raw.reconfigureRawStream(rawSensor.format, rawSensor.width, rawSensor.height)
            }
            running = false; logger.flush()
        }
        onStatus(if (productionMode) "RayNeo SoS · run complete" else "EXP5.5 done")
    }

    /** Formation, inference, and fallback-metric cost at each K the controller actually uses,
     *  followed by a sensor-control lag sweep.
     *
     *  `formation_ms` here excludes capture, since the frames are taken once up front and
     *  reused. The identically-named column in frames.csv DOES include capture — the two are
     *  not comparable. */
    fun runBench(onStatus: (String) -> Unit) {
        writeManifest("bench", isGtReference = false, captureWidth = -1, captureHeight = -1)
        logger.csv("bench", listOf("k", "imgsz", "quant", "batch_mode",
            "formation_ms", "infer_ms_p50", "infer_ms_p95",
            "entropy_ms_p50", "entropy_ms_p95",
            "laplacian_ms_p50", "laplacian_ms_p95",
            "tenengrad_ms_p50", "tenengrad_ms_p95",
            "crete_ms_p50", "crete_ms_p95"))
        running = true
        val frames = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        if (frames.isEmpty()) { logger.flush(); return }
        // Candidate sets must mirror what plan() emits, NOT plain 0..K-1. The two axes cost
        // very different amounts in formAllCells — the shutter axis needs a burst-sum and a
        // demosaic per ROW, the gain axis is only a per-cell multiply — and since
        // cell = gi*nShutter + sj, numbering 0..K-1 walks the expensive axis.
        val cellSets = listOf(
            intArrayOf(grid.cell(0, 0)),                    // K=1     single cell
            IntArray(grid.nGain) { grid.cell(it, 0) },      // K=nGain off-probe: gain COLUMN
            IntArray(grid.nGain * grid.nShutter) { it }     // K=N     probe step: full grid
        ).distinctBy { it.size }
        for (cells in cellSets) {
            if (!running) break
            val k = cells.size
            val tf = now(); val imgs = formCells(frames, cells); val fms = ms(tf)
            val infers = ArrayList<Double>()
            repeat(5) { val ti = now(); detector.detectBatch(imgs); infers.add(ms(ti)) }
            // Same call shape as the Proposed tie-breaker, so these are the real cost added to
            // a step where the fallback fires. All four are timed to compare them side by side.
            fun timeMetric(metric: FallbackMetric): Pair<Double, Double> {
                val times = ArrayList<Double>(5)
                repeat(5) { val t = now(); fallbackScores(imgs, metric); times.add(ms(t)) }
                times.sort()
                val p50 = times[times.size / 2]
                val p95 = times[(times.size * 95 / 100).coerceAtMost(times.size - 1)]
                return p50 to p95
            }
            val (entP50, entP95) = timeMetric(FallbackMetric.ENTROPY)
            val (lapP50, lapP95) = timeMetric(FallbackMetric.LAPLACIAN_VAR)
            val (tenP50, tenP95) = timeMetric(FallbackMetric.TENENGRAD_NORM)
            val (creP50, creP95) = timeMetric(FallbackMetric.CRETE_ROFFET)
            infers.sort()
            val infP50 = infers[infers.size / 2]
            val infP95 = infers[(infers.size * 95 / 100).coerceAtMost(infers.size - 1)]
            logger.row("bench", listOf(k, 640, "fp16", "loop",
                "%.1f".format(fms), "%.1f".format(infP50), "%.1f".format(infP95),
                "%.1f".format(entP50), "%.1f".format(entP95),
                "%.1f".format(lapP50), "%.1f".format(lapP95),
                "%.1f".format(tenP50), "%.1f".format(tenP95),
                "%.1f".format(creP50), "%.1f".format(creP95)))
            onStatus("Bench K=$k infer=${"%.0f".format(infP50)}ms ent=${"%.0f".format(entP50)} lap=${"%.0f".format(lapP50)} ten=${"%.0f".format(tenP50)} cre=${"%.0f".format(creP50)}")
        }

        // Sensor control lag: cycle ISO across the grid and record wall-clock per capture,
        // whether the first result after each change already carries the requested value
        // (an empirical check on SYNC_MAX_LATENCY), and sensor timestamps for clock alignment.
        logger.csv("lag", listOf("step", "iso_req", "iso_applied",
            "exp_req_us", "exp_applied_us", "frame_number", "sensor_ts_ns", "wall_ms"))
        val gainsSorted = grid.gains.toList().distinct().sorted()
        val lo = gainsSorted.first()
        val mid = gainsSorted[gainsSorted.size / 2]
        val hi = gainsSorted.last()
        // Covers every pairwise transition plus a repeated value, which gives the baseline
        // pipeline time with no setting change at all.
        val isoSeq = listOf(lo, hi, lo, mid, hi, mid, lo, lo)
        val fixedExpUs = grid.exposuresUs[0]
        for ((step, iso) in isoSeq.withIndex()) {
            if (!running) break
            val tNs = System.nanoTime()
            val frames = raw.capture(fixedExpUs, iso, 1)
            val wallMs = (System.nanoTime() - tNs) / 1e6
            if (frames.isEmpty()) continue
            val m = raw.lastMeta.firstOrNull()
            logger.row("lag", listOf(step, iso, m?.appliedIso ?: -1,
                fixedExpUs, m?.appliedExpUs ?: -1L,
                m?.frameNumber ?: -1L, m?.timestamp ?: -1L,
                "%.1f".format(wallMs)))
            onStatus("Lag step=$step iso=$iso applied=${m?.appliedIso} ${"%.0f".format(wallMs)}ms")
        }
        logger.flush()
    }

    // ---------- shared helpers ----------

    /** Writes the manifest, opens the shared logs, and arms [running]. Modes with their own
     *  row schema pass `withFrames = false`. */
    private fun startPass(method: String, isGtReference: Boolean,
                          withFrames: Boolean = true,
                          methodParams: JSONObject? = null) {
        passStartMs = System.currentTimeMillis()
        val w = if (raw.captureWidth > 0) raw.captureWidth else -1
        val h = if (raw.captureHeight > 0) raw.captureHeight else -1
        writeManifest(method, isGtReference, w, h, methodParams)
        if (withFrames) logger.csv("frames", headers())
        wireImuLog()
        detectionTotalAboveThresh = 0
        detectionTotalAtFloor = 0
        totalFramesLogged = 0
        running = true
    }

    private fun endPass() {
        unwireImuLog()
        logger.flush()
        // Written so the run's totals can be cited from the logs alone.
        runCatching {
            val summary = JSONObject()
                .put("total_frames_logged", totalFramesLogged)
                .put("total_dets_above_threshold", detectionTotalAboveThresh)
                .put("total_dets_at_floor", detectionTotalAtFloor)
                .put("select_conf_threshold", selectConf)
                .put("detector_conf_floor", 0.01)
                .put("ts_ms", System.currentTimeMillis())
            java.io.File(logger.dir, "summary.json").writeText(summary.toString(2))
        }
    }

    private fun frameName(idx: Int) = "frame_%04d".format(idx)

    /** Mean luma of a formed bitmap as a fraction of white, for SAFE_CELL's brightness target. */
    private fun meanLumaRatio(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        val step = (px.size / 100_000).coerceAtLeast(1)
        var sum = 0L; var count = 0
        var i = 0
        while (i < px.size) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            sum += (r * 299 + g * 587 + b * 114) / 1000
            count++
            i += step
        }
        return if (count == 0) 0.0 else (sum.toDouble() / count) / 255.0
    }

    /** Mean RAW value as a fraction of white, for Custom AE's brightness feedback. */
    private fun meanRawRatio(frame: RawFrame): Double {
        val pixels = frame.bayer
        val step = (pixels.size / 100_000).coerceAtLeast(1)
        var sum = 0L; var count = 0
        var i = 0
        while (i < pixels.size) { sum += pixels[i].toLong(); count++; i += step }
        return if (count == 0 || frame.maxDn <= 0) 0.0
        else (sum.toDouble() / count) / frame.maxDn
    }

    /** Closed-loop AE driving mean brightness toward [targetRatio]. ISO absorbs the correction
     *  first and exposure only grows once ISO clamps, which keeps motion blur down. Unlike
     *  phone AE it is fully determined by the brightness sequence, so runs are reproducible. */
    private class CustomAeBrightness(
        val targetRatio: Double = 0.40,    // bright but short of clipping
        val isoMin: Int = 100,
        val isoMax: Int = 1600,
        val expMinUs: Int = 1000,
        val expMaxUs: Int = 32_000,        // handheld motion-blur cap
        val maxStep: Double = 4.0          // per-call scale limit, damps dim -> bright swings
    ) {
        fun next(currentIso: Int, currentExpUs: Int, lastRatio: Double): Pair<Int, Int> {
            if (lastRatio < 0.001) return Pair(currentIso, currentExpUs)  // no signal — hold
            val scale = (targetRatio / lastRatio).coerceIn(1.0 / maxStep, maxStep)
            val newIso = (currentIso * scale).toInt().coerceIn(isoMin, isoMax)
            val resScale = scale * currentIso / newIso       // what ISO could not absorb
            val newExp = (currentExpUs * resScale).toInt().coerceIn(expMinUs, expMaxUs)
            return Pair(newIso, newExp)
        }

        fun toJson(): JSONObject = JSONObject()
            .put("target_ratio", targetRatio)
            .put("iso_min", isoMin).put("iso_max", isoMax)
            .put("exp_min_us", expMinUs).put("exp_max_us", expMaxUs)
            .put("max_step", maxStep)
    }

    /** ISO + exposure tag for filenames. Milliseconds when the exposure is whole, microseconds
     *  otherwise, so it is always lossless. No dots — they confuse extension parsing. */
    private fun isoExpTag(iso: Int, expUs: Int): String =
        if (expUs % 1000 == 0) "iso${iso}_exp${expUs / 1000}ms"
        else "iso${iso}_exp${expUs}us"

    /** Physical ISO × digitalBoost. Every bitmap goes through a formation path that applies
     *  the boost, so this is the number that matches what a JPEG actually looks like, and it
     *  is what JPEG filenames and the cell-grid UI use. RAW files keep physical ISO. */
    private fun effIso(iso: Int): Int = (iso * grid.digitalBoost).toInt()

    /** One bitmap from one capture that was already taken at the requested cell, so no digital
     *  re-gain beyond digitalBoost. Goes through the parallel path for the LUT and worker pool. */
    private fun bitmapFromRaw(frame: RawFrame): Bitmap =
        parallelFormation.formAllCells(listOf(frame), intArrayOf(grid.cell(0, 0)))[0]

    /** Shared formation path for the modes that capture separately from forming. Same
     *  implementation Proposed uses, so Bench measures the deployed cost. */
    private val parallelFormation by lazy {
        ParallelRawCandidateSource(grid, raw, colorPipeline = colorPipeline)
    }
    private fun formCells(frames: List<RawFrame>, cells: IntArray): List<Bitmap> =
        parallelFormation.formAllCells(frames, cells)
}
