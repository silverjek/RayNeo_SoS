package com.example.activeperception.acquire

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.TimeUnit

/** Pre-letterboxed RGB float tensor plus the geometry needed to map detections back. */
data class DirectTensorBatch(
    val input: ByteBuffer,
    val transforms: List<TensorLetterbox>
)

data class TensorLetterbox(val scale: Double, val padX: Double, val padY: Double)

data class TensorSnapshot(val values: FloatArray, val transforms: List<TensorLetterbox>)

/**
 * `Detector<Bitmap>` via TensorFlow Lite with the Adreno GPU (OpenCL) delegate, falling
 * back to CPU/XNNPACK if the delegate can't be built.
 *
 * True batching needs a fixed batch dim, so one Interpreter is loaded per batch size and
 * [detectBatch] picks the smallest one that fits. For K ∈ {3, 9} — the gain-column and
 * full-grid steps — the GPU runs all candidates in one launch, several times faster on
 * Adreno than looping. The batched assets are exported from `tools/yolov8n_640_dyn.onnx`,
 * an Ultralytics export patched to use Reshape `0` so the batch dim propagates.
 *
 * Every slot's output is [4 + numClasses, 8400] per item, so the decode path is shared.
 */
class TfliteYoloDetector(
    context: Context,
    /** (asset, batchSize) in ascending batch order; inputs larger than the last entry chunk. */
    batchedAssets: List<Pair<String, Int>> = listOf(
        "yolov8n_640_fp16.tflite" to 1,
        "yolov8n_640_b3_fp16.tflite" to 3,
        "yolov8n_640_b9_fp16.tflite" to 9
    ),
    private val imgsz: Int = 640,
    private val confThresh: Float = 0.01f,
    private val iouThresh: Double = 0.45,
    private val maxDet: Int = 100,
    // null keeps every class emitted by the COCO-80 model. The original phone experiment
    // restricted selection to vehicle classes {car, motorcycle, bus, truck}; RayNeo is a
    // general first-person camera, so its default experiment domain is all COCO classes.
    private val allowed: Set<Int>? = null,
    private val numClasses: Int = 80,
    /** Optional local model-output class index -> COCO class index mapping. */
    private val classIdMap: IntArray? = null,
    private val accelerator: Accelerator = Accelerator.GPU,
    /** Probe mode can disable fallback so a CPU result is never mistaken for GPU success. */
    private val allowFallback: Boolean = true,
    /** The paper requires a GPU delegate but not a specific Android GPU API. */
    private val gpuBackend: GpuBackend = GpuBackend.OPENCL,
    /** Current Ultralytics Full-INT8 LiteRT exports emit normalized xywh. */
    private val normalizedOutputBoxes: Boolean = false,
    /** Reports long per-batch delegate preparation without changing detection behaviour. */
    private val onLoadStatus: ((String) -> Unit)? = null
) : Detector<Bitmap> {

    enum class Accelerator { GPU, NNAPI, CPU }
    enum class GpuBackend { AUTO, OPENGL, OPENCL }
    companion object {
        private const val TAG = "TfliteYoloDetector"
        val COCO5_CLASS_IDS = intArrayOf(41, 40, 46, 5, 60)
    }

    private val appContext = context.applicationContext

    /** One Interpreter pinned at a batch size, with its dedicated I/O buffers.
     *  The buffers are reused across calls, so [detectBatch] is NOT thread-safe. */
    private class BatchSlot(
        val batch: Int,
        val interp: Interpreter,
        val gpuDelegate: GpuDelegate?,
        val backend: String,
        val nchwInput: Boolean,
        val anchors: Int,
        val input: ByteBuffer,
        val output: Array<Array<FloatArray>>
    )

    private val slots: List<BatchSlot>
    private val preprocPool = Executors.newFixedThreadPool(4)
    private val decodePool = Executors.newFixedThreadPool(4)
    private val numAnchors = (imgsz / 8) * (imgsz / 8) +
        (imgsz / 16) * (imgsz / 16) + (imgsz / 32) * (imgsz / 32)
    private val optimizedDecoder = OptimizedYoloDecode(
        numClasses, numAnchors, confThresh, allowed, iouThresh, maxDet, preNmsTopK = 1000,
        classIdMap = classIdMap)
    private val coco5Decoder = if (numClasses == 80) OptimizedYoloDecode(
        numClasses, numAnchors, confThresh, allowed, iouThresh, maxDet, preNmsTopK = 1000,
        classIndices = COCO5_CLASS_IDS) else null
    private fun outputTransform(transform: TensorLetterbox): TensorLetterbox =
        if (normalizedOutputBoxes) TensorLetterbox(
            transform.scale / imgsz, transform.padX / imgsz, transform.padY / imgsz)
        else transform
    private data class PreprocessScratch(
        val bitmap: Bitmap,
        val canvas: Canvas,
        val pixels: IntArray,
        val dst: RectF,
        val paint: Paint
    )
    private val scratchBitmaps = ConcurrentLinkedQueue<Bitmap>()
    private val preprocessScratch = ThreadLocal.withInitial {
        val bitmap = Bitmap.createBitmap(imgsz, imgsz, Bitmap.Config.ARGB_8888)
        scratchBitmaps.add(bitmap)
        PreprocessScratch(bitmap, Canvas(bitmap), IntArray(imgsz * imgsz), RectF(),
            Paint(Paint.FILTER_BITMAP_FLAG))
    }

    /** Last detectBatch breakdown (preprocess / GPU run / decode ms). */
    @Volatile var lastPreprocessMs: Double = 0.0; private set
    @Volatile var lastRunMs: Double = 0.0; private set
    @Volatile var lastDecodeMs: Double = 0.0; private set
    @Volatile var lastSlotBatch: Int = 0; private set
    @Volatile var lastPreNmsCandidates: Int = -1; private set
    @Volatile var lastTopKCandidates: Int = -1; private set
    val backendSummary: String get() = slots.joinToString(",") { "B${it.batch}:${it.backend}" }

    /** Force lazy GPU compilation/allocation for every deployed batch before a gated run. */
    fun warmUpAllBatches() {
        val neutral = Bitmap.createBitmap(imgsz, imgsz, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.rgb(114, 114, 114))
        }
        try {
            for (slot in slots) detectBatch(List(slot.batch) { neutral })
        } finally {
            neutral.recycle()
        }
    }

    init {
        slots = batchedAssets.sortedBy { it.second }.mapNotNull { (asset, batch) ->
            onLoadStatus?.invoke("preparing ${accelerator.name} B=$batch…")
            runCatching { loadSlotWithFallback(context, asset, batch, accelerator) }
                .onFailure { Log.w(TAG, "skip $asset (B=$batch): ${it.message}") }
                .getOrNull().also { slot ->
                    onLoadStatus?.invoke(if (slot != null) "${slot.backend} B=$batch ready"
                        else "B=$batch unavailable")
                }
        }
        require(slots.isNotEmpty()) { "no TFLite interpreter could be loaded" }
        Log.d(TAG, "loaded batches=${slots.map { it.batch }}")
    }

    private fun loadSlotWithFallback(
        context: Context, asset: String, batch: Int, preferred: Accelerator
    ): BatchSlot {
        val order = if (!allowFallback) listOf(preferred) else when (preferred) {
            Accelerator.GPU -> listOf(Accelerator.GPU, Accelerator.NNAPI, Accelerator.CPU)
            Accelerator.NNAPI -> listOf(Accelerator.NNAPI, Accelerator.CPU)
            Accelerator.CPU -> listOf(Accelerator.CPU)
        }
        var last: Throwable? = null
        for (candidate in order) {
            try {
                return loadSlot(context, asset, batch, candidate)
            } catch (error: Throwable) {
                last = error
                Log.w(TAG, "$asset (B=$batch) ${candidate.name} unavailable", error)
            }
        }
        throw IllegalStateException("No backend for $asset", last)
    }

    private fun loadSlot(context: Context, asset: String, batch: Int, accel: Accelerator): BatchSlot {
        val bytes = context.assets.open(asset).use { it.readBytes() }
        val buf = ByteBuffer.allocateDirect(bytes.size).apply {
            order(ByteOrder.nativeOrder()); put(bytes); rewind()
        }
        val opts = Interpreter.Options()
        val gpu: GpuDelegate? = when (accel) {
            Accelerator.GPU -> {
                val gpuOpts = GpuDelegateFactory.Options().apply {
                    when (gpuBackend) {
                        GpuBackend.AUTO -> Unit
                        GpuBackend.OPENGL -> setForceBackend(
                            GpuDelegateFactory.Options.GpuBackend.OPENGL)
                        GpuBackend.OPENCL -> setForceBackend(
                            GpuDelegateFactory.Options.GpuBackend.OPENCL)
                    }
                    setQuantizedModelsAllowed(true)
                    setInferencePreference(
                        GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                    setSerializationParams(
                        context.cacheDir.absolutePath, "$asset.${gpuBackend.name.lowercase()}.v2")
                }
                GpuDelegate(gpuOpts).also {
                    opts.addDelegate(it)
                    Log.d(TAG, "$asset (B=$batch): GPU ${gpuBackend.name}")
                }
            }
            Accelerator.NNAPI -> { @Suppress("DEPRECATION") opts.setUseNNAPI(true); null }
            Accelerator.CPU -> { opts.setNumThreads(4); null }
        }
        val interp = try {
            Interpreter(buf, opts)
        } catch (error: Throwable) {
            runCatching { gpu?.close() }
            throw error
        }
        val inputTensor = interp.getInputTensor(0)
        val inputShape = inputTensor.shape()
        require(inputTensor.dataType() == DataType.FLOAT32) {
            "$asset input must expose FLOAT32 boundary, got ${inputTensor.dataType()}"
        }
        val nchw = inputShape.contentEquals(intArrayOf(batch, 3, imgsz, imgsz))
        val nhwc = inputShape.contentEquals(intArrayOf(batch, imgsz, imgsz, 3))
        require(nchw || nhwc) { "$asset unsupported input shape=${inputShape.contentToString()}" }
        val outputTensor = interp.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        require(outputTensor.dataType() == DataType.FLOAT32) {
            "$asset output must expose FLOAT32 boundary, got ${outputTensor.dataType()}"
        }
        require(outputShape.contentEquals(intArrayOf(batch, 4 + numClasses, numAnchors))) {
            "$asset output=${outputShape.contentToString()}, expected=[$batch,${4 + numClasses},$numAnchors]"
        }
        val input = ByteBuffer.allocateDirect(batch * imgsz * imgsz * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        val output = Array(batch) { Array(4 + numClasses) { FloatArray(numAnchors) } }
        Log.d(TAG, "$asset tensors input=${inputShape.contentToString()} output=${outputShape.contentToString()}")
        return BatchSlot(batch, interp, gpu, accel.name, nchw, numAnchors, input, output)
    }

    override fun detectBatch(images: List<Bitmap>): List<List<Detection>> {
        if (images.isEmpty()) return emptyList()
        // Smallest slot that fits, else the largest and chunk. A size that falls between two
        // slots pays the larger slot's full cost (the unused lanes are zero-padded).
        val slot = slots.firstOrNull { it.batch >= images.size } ?: slots.last()
        val out = ArrayList<List<Detection>>(images.size)
        var i = 0
        while (i < images.size) {
            val end = minOf(i + slot.batch, images.size)
            out.addAll(detectChunk(images.subList(i, end), slot))
            i = end
        }
        return out
    }

    private fun detectChunk(chunk: List<Bitmap>, slot: BatchSlot): List<List<Detection>> {
        val tPre = System.nanoTime()
        val frameBytes = imgsz * imgsz * 3 * 4
        // Workers write to disjoint slices of the shared DirectByteBuffer via absolute-index
        // puts, so there is no shared position state to contend on.
        val futures = chunk.mapIndexed { i, bmp ->
            preprocPool.submit<TensorLetterbox> {
                preprocessAt(slot.input, bmp, i * frameBytes, slot.nchwInput)
            }
        }
        val pps = futures.map { it.get() }
        // Zero the unused lanes so the GPU sees deterministic data.
        for (i in chunk.size until slot.batch) {
            val base = i * frameBytes
            for (k in 0 until imgsz * imgsz * 3) slot.input.putFloat(base + k * 4, 0f)
        }
        slot.input.rewind()

        val tRun = System.nanoTime()
        slot.interp.run(slot.input, slot.output)

        val tDec = System.nanoTime()
        val out = ArrayList<List<Detection>>(chunk.size)
        for (i in chunk.indices) {
            val pp = pps[i]
            val flat = FloatArray((4 + numClasses) * slot.anchors)
            for (c in 0 until 4 + numClasses) {
                System.arraycopy(slot.output[i][c], 0, flat, c * slot.anchors, slot.anchors)
            }
            val dets = YoloDecode.decode(
                flat, numClasses, slot.anchors, confThresh, allowed, iouThresh, maxDet)
            val decodeMap = outputTransform(pp)
            out.add(YoloDecode.unletterbox(dets, decodeMap.scale, decodeMap.padX, decodeMap.padY))
        }
        val tEnd = System.nanoTime()

        lastPreprocessMs = (tRun - tPre) / 1e6
        lastRunMs = (tDec - tRun) / 1e6
        lastDecodeMs = (tEnd - tDec) / 1e6
        lastSlotBatch = slot.batch
        Log.d(TAG, "B=${slot.batch} chunk=${chunk.size}  pre=${"%.1f".format(lastPreprocessMs)}  run=${"%.1f".format(lastRunMs)}  dec=${"%.1f".format(lastDecodeMs)} ms")

        return out
    }

    /** EXP2.1-only entry point: inference over an RGB float tensor that was letterboxed
     *  directly from the demosaiced RGB planes. Production [detectBatch] is unchanged. */
    fun detectTensorBatch(batch: DirectTensorBatch): List<List<Detection>> {
        val count = batch.transforms.size
        if (count == 0) return emptyList()
        val slot = slots.firstOrNull { it.batch == count }
            ?: error("EXP2.1 requires an exact B=$count interpreter")
        val expected = slot.batch * imgsz * imgsz * 3 * 4
        require(batch.input.capacity() >= expected) {
            "tensor buffer ${batch.input.capacity()} < expected $expected"
        }
        batch.input.rewind()
        val tRun = System.nanoTime()
        slot.interp.run(batch.input, slot.output)
        val tDec = System.nanoTime()
        val out = decodeSlot(slot, count, batch.transforms)
        val tEnd = System.nanoTime()
        lastPreprocessMs = 0.0
        lastRunMs = (tDec - tRun) / 1e6
        lastDecodeMs = (tEnd - tDec) / 1e6
        lastSlotBatch = slot.batch
        return out
    }

    /** EXP2.2 treatment path. GPU execution is unchanged; only output decode/NMS differs. */
    fun detectTensorBatchOptimized(batch: DirectTensorBatch): List<List<Detection>> =
        detectTensorBatchWithDecoder(batch, optimizedDecoder, "EXP2.2")

    /** Production-safe fixed-batch execution. RayNeo's clean-cache B=9 OpenCL delegate can
     * block indefinitely during construction, so a K=9 tensor is processed as three proven
     * B=3 launches when no exact B=9 slot is loaded. Input tensors remain contiguous NHWC;
     * each chunk is copied with a direct-buffer bulk put rather than per-float conversion. */
    fun detectTensorBatchOptimizedFlexible(batch: DirectTensorBatch): List<List<Detection>> {
        val count = batch.transforms.size
        if (slots.any { it.batch == count }) return detectTensorBatchOptimized(batch)
        val slot = slots.filter { it.batch < count }.maxByOrNull { it.batch }
            ?: error("No batch slot can process K=$count")
        require(count % slot.batch == 0) { "K=$count is not divisible by B=${slot.batch}" }
        val bytesPerLane = imgsz * imgsz * 3 * 4
        val out = ArrayList<List<Detection>>(count)
        var totalRun = 0.0; var totalDecode = 0.0
        var offset = 0
        while (offset < count) {
            val src = batch.input.duplicate().order(ByteOrder.nativeOrder())
            src.position(offset * bytesPerLane)
            src.limit((offset + slot.batch) * bytesPerLane)
            slot.input.clear(); slot.input.put(src); slot.input.rewind()
            out += detectTensorBatchWithDecoder(
                DirectTensorBatch(slot.input, batch.transforms.subList(offset, offset + slot.batch)),
                optimizedDecoder, "FINAL_CHUNKED_B${slot.batch}")
            totalRun += lastRunMs; totalDecode += lastDecodeMs
            offset += slot.batch
        }
        lastPreprocessMs = 0.0
        lastRunMs = totalRun
        lastDecodeMs = totalDecode
        lastSlotBatch = slot.batch
        return out
    }

    /** EXP2.3 arm 1: retain the 80-class model output but scan only five COCO channels. */
    fun detectTensorBatchOptimizedCoco5(batch: DirectTensorBatch): List<List<Detection>> =
        detectTensorBatchWithDecoder(batch, requireNotNull(coco5Decoder), "EXP2.3")

    private fun detectTensorBatchWithDecoder(
        batch: DirectTensorBatch, decoder: OptimizedYoloDecode, experiment: String
    ): List<List<Detection>> {
        val count = batch.transforms.size
        if (count == 0) return emptyList()
        val slot = slots.firstOrNull { it.batch == count }
            ?: error("$experiment requires an exact B=$count interpreter")
        val expected = slot.batch * imgsz * imgsz * 3 * 4
        require(batch.input.capacity() >= expected)
        batch.input.rewind()
        val tRun = System.nanoTime()
        slot.interp.run(batch.input, slot.output)
        val tDec = System.nanoTime()
        // Each worker reads one immutable output lane and owns ThreadLocal primitive scratch.
        val decoded = if (count == 1) {
            // A pool round trip is pure overhead when there is no batch-lane parallelism.
            listOf(decoder.decode(slot.output[0], outputTransform(batch.transforms[0])))
        } else {
            val futures = (0 until count).map { lane ->
                decodePool.submit<OptimizedDecodeResult> {
                    decoder.decode(slot.output[lane], outputTransform(batch.transforms[lane]))
                }
            }
            futures.map { it.get() }
        }
        val tEnd = System.nanoTime()
        lastPreprocessMs = 0.0
        lastRunMs = (tDec - tRun) / 1e6
        lastDecodeMs = (tEnd - tDec) / 1e6
        lastSlotBatch = slot.batch
        lastPreNmsCandidates = decoded.sumOf { it.preNmsCandidates }
        lastTopKCandidates = decoded.sumOf { it.topKCandidates }
        return decoded.map { it.detections }
    }

    /** One fixed-batch five-output detector at a time limits peak GPU memory in EXP2.3. */
    fun createCoco5HeadDetector(asset: String, batch: Int): TfliteYoloDetector =
        TfliteYoloDetector(
            appContext, listOf(asset to batch), imgsz, confThresh, iouThresh, maxDet,
            allowed, numClasses = 5, classIdMap = COCO5_CLASS_IDS,
            accelerator = accelerator, allowFallback = false, gpuBackend = gpuBackend)

    /** One-off equivalence probe used outside timed repetitions. It exposes the exact float
     *  values produced by the current Bitmap/Canvas preprocessing path. */
    fun snapshotBitmapTensor(images: List<Bitmap>): TensorSnapshot {
        require(images.isNotEmpty())
        val slot = slots.firstOrNull { it.batch == images.size }
            ?: error("EXP2.1 requires an exact B=${images.size} interpreter")
        val frameBytes = imgsz * imgsz * 3 * 4
        val futures = images.mapIndexed { i, bmp ->
            preprocPool.submit<TensorLetterbox> {
                preprocessAt(slot.input, bmp, i * frameBytes, slot.nchwInput)
            }
        }
        val mappings = futures.map { it.get() }
        val values = FloatArray(images.size * imgsz * imgsz * 3)
        for (i in values.indices) values[i] = slot.input.getFloat(i * 4)
        return TensorSnapshot(values, mappings)
    }

    private fun decodeSlot(
        slot: BatchSlot, count: Int, transforms: List<TensorLetterbox>
    ): List<List<Detection>> {
        val out = ArrayList<List<Detection>>(count)
        for (i in 0 until count) {
            val pp = transforms[i]
            val flat = FloatArray((4 + numClasses) * slot.anchors)
            for (c in 0 until 4 + numClasses) {
                System.arraycopy(slot.output[i][c], 0, flat, c * slot.anchors, slot.anchors)
            }
            val dets = YoloDecode.decode(
                flat, numClasses, slot.anchors, confThresh, allowed, iouThresh, maxDet)
            val decodeMap = outputTransform(pp)
            out.add(YoloDecode.unletterbox(dets, decodeMap.scale, decodeMap.padX, decodeMap.padY))
        }
        return out
    }

    /** Aspect-preserving letterbox to imgsz², written as RGB HWC float [0,1] into [dst]
     *  at absolute byte offset [baseOffset]. Returns the mapping needed to unletterbox. */
    private fun preprocessAt(
        dst: ByteBuffer, bmp: Bitmap, baseOffset: Int, nchw: Boolean
    ): TensorLetterbox {
        val scale = minOf(imgsz.toFloat() / bmp.width, imgsz.toFloat() / bmp.height)
        val nw = Math.round(bmp.width * scale); val nh = Math.round(bmp.height * scale)
        val padX = (imgsz - nw) / 2f; val padY = (imgsz - nh) / 2f
        // One fixed scratch raster per worker. The previous path allocated two Bitmaps and
        // one IntArray per candidate (scaled image + canvas + pixels), which drove PSS above
        // 500 MB during B=9 experiments and increased crash risk. Drawing into the final
        // letterbox raster directly preserves the same filtered resize and normalization.
        val scratch = requireNotNull(preprocessScratch.get())
        scratch.canvas.drawColor(Color.rgb(114, 114, 114))
        scratch.dst.set(padX, padY, padX + nw, padY + nh)
        scratch.canvas.drawBitmap(bmp, null, scratch.dst, scratch.paint)
        scratch.bitmap.getPixels(scratch.pixels, 0, imgsz, 0, 0, imgsz, imgsz)
        if (nchw) {
            val planeBytes = imgsz * imgsz * 4
            for (i in scratch.pixels.indices) {
                val p = scratch.pixels[i]
                dst.putFloat(baseOffset + i * 4, ((p ushr 16) and 0xFF) / 255f)
                dst.putFloat(baseOffset + planeBytes + i * 4, ((p ushr 8) and 0xFF) / 255f)
                dst.putFloat(baseOffset + planeBytes * 2 + i * 4, (p and 0xFF) / 255f)
            }
        } else {
            var pos = baseOffset
            for (p in scratch.pixels) {
                dst.putFloat(pos, ((p ushr 16) and 0xFF) / 255f); pos += 4
                dst.putFloat(pos, ((p ushr 8) and 0xFF) / 255f); pos += 4
                dst.putFloat(pos, (p and 0xFF) / 255f); pos += 4
            }
        }
        return TensorLetterbox(scale.toDouble(), padX.toDouble(), padY.toDouble())
    }

    fun close() {
        slots.forEach {
            runCatching { it.interp.close() }
            runCatching { it.gpuDelegate?.close() }
        }
        runCatching { preprocPool.shutdown(); preprocPool.awaitTermination(2, TimeUnit.SECONDS) }
        runCatching { decodePool.shutdown(); decodePool.awaitTermination(2, TimeUnit.SECONDS) }
        scratchBitmaps.forEach { if (!it.isRecycled) it.recycle() }
        scratchBitmaps.clear()
    }
}
