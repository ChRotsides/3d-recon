package com.recon.videoscan.live

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.os.Build
import android.os.SystemClock
import android.system.Os
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.lang.reflect.InvocationTargetException
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.nio.file.FileVisitResult
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.SimpleFileVisitor
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import java.util.Properties
import java.util.UUID

/**
 * UniDepth v2 ViT-S at its original 182x322 input geometry, with the network on the Hexagon NPU.
 * Both files come from video-recon/tools/qnn/prepare_unidepth_qnn.py:
 *  - [CAMERA_MODEL] maps this frame's K [1,3,3] to the camera-conditioning tensors on the CPU.
 *    It runs for every frame; nothing derived from K is reused between calls.
 *  - [MAIN_MODEL] maps image [1,3,182,322] plus those tensors to metric depth [1,182,322] on QNN HTP.
 *    CPU fallback is disabled, so a graph the NPU cannot take whole fails here instead of running slowly.
 *
 * The compiled QNN context is cached under [cacheDirectory] per model, runtime, provider options and
 * device. A cold cache compiles for minutes (203.7 s measured on SM-S918B): construct on a background
 * thread and keep a CPU depth model meanwhile. Needs the onnxruntime-android-qnn package; with plain
 * onnxruntime-android the constructor throws [UnsupportedOperationException].
 */
class QnnDepthSession @JvmOverloads constructor(
    modelsDir: File,
    cacheDir: File,
    nativeLibDir: String,
    /** QNN "htp_arch" provider option, e.g. "73"; null lets the backend use the device it runs on. */
    htpArch: String? = null,
) : AutoCloseable {
    private class MainSession(val session: OrtSession, val fromCache: Boolean, val compileMillis: Long)
    private class Opened(val camera: OrtSession, val cameraOutputs: List<String>, val main: MainSession)

    private val environment = OrtEnvironment.getEnvironment()
    private val camera: OrtSession
    private val cameraOutputs: List<String>
    private val main: OrtSession
    private var closed = false

    /** True when the NPU graph came from a validated cached context, without compiling. */
    val loadedFromCache: Boolean
    /** Cache entry name: SHA-256 prefix of the model, runtime libraries, provider options and device. */
    val cacheKey: String
    /** Wall time spent compiling the QNN context during construction; 0 when loaded from cache. */
    val compileMillis: Long

    init {
        val mainFile = File(modelsDir, MAIN_MODEL)
        val cameraFile = File(modelsDir, CAMERA_MODEL)
        require(mainFile.isFile && mainFile.length() > 0) { "Missing QNN depth model $mainFile" }
        require(cameraFile.isFile && cameraFile.length() > 0) { "Missing QNN camera model $cameraFile" }
        require(htpArch == null || htpArch.matches(Regex("[0-9]{2}"))) { "Invalid htp_arch $htpArch" }
        requireQnnApi()
        val qnnOptions = sortedMapOf("backend_path" to "libQnnHtp.so", "enable_htp_fp16_precision" to "1",
            "htp_performance_mode" to "burst", "htp_graph_finalization_optimization_mode" to "1")
        if (htpArch != null) qnnOptions["htp_arch"] = htpArch
        configureDspLibraryPath(nativeLibDir)
        val identity = cacheIdentity(environment, mainFile, nativeLibDir, qnnOptions)
        cacheKey = hex(sha256(identity.toByteArray(Charsets.UTF_8))).take(32)
        val opened = openSessions(environment, cameraFile, mainFile, cacheDirectory(cacheDir), cacheKey, identity, qnnOptions)
        camera = opened.camera; cameraOutputs = opened.cameraOutputs; main = opened.main.session
        loadedFromCache = opened.main.fromCache; compileMillis = opened.main.compileMillis
        Log.i(TAG, "QNN depth ready key=$cacheKey fromCache=$loadedFromCache compile=${compileMillis}ms")
    }

    /** [chw]: ImageNet-normalised RGB planes, 3 x [HEIGHT] x [WIDTH]. [k]: fx, fy, cx, cy at that size.
     * Returns metric depth, [HEIGHT] x [WIDTH], row-major. */
    @Synchronized fun run(chw: FloatArray, k: FloatArray): FloatArray {
        check(!closed) { "QnnDepthSession is closed" }
        return infer(environment, camera, cameraOutputs, main, chw, k)
    }

    @Synchronized override fun close() {
        if (closed) return
        closed = true
        try { main.close() } finally { camera.close() }
    }

    companion object {
        const val MAIN_MODEL = "qnn_main_static_gelu.onnx"
        const val CAMERA_MODEL = "qnn_camera.onnx"
        const val WIDTH = 322
        const val HEIGHT = 182
        /** Created inside the caller's cache directory; nothing outside it is written or removed. */
        const val CACHE_DIRECTORY = "qnn-depth-context-v1"

        private const val TAG = "livescan"
        private const val CONTEXT_FILE = "main_ctx.onnx"
        private const val MANIFEST_FILE = "manifest.properties"
        private const val LOCK_FILE = ".lock"
        private const val QUARANTINE_DIRECTORY = "quarantine"
        private const val MAX_QUARANTINED = 2
        private const val IMAGE_INPUT = "image"
        private const val K_INPUT = "K"
        private const val DEPTH_OUTPUT = "depth"
        private const val ADSP_LIBRARY_PATH = "ADSP_LIBRARY_PATH"
        private val VENDOR_DSP_PATHS = listOf("/vendor/lib/rfsa/adsp", "/vendor/dsp/cdsp", "/system/lib/rfsa/adsp",
            "/system/vendor/lib/rfsa/adsp", "/dsp")
        private val BUILD_NAME = Regex("\\.build-[0-9a-f]{32}-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
        private val QUARANTINE_NAME = Regex("[0-9a-f]{32}-[0-9]+")
        private val NATIVE_LIBRARY = Regex("lib(onnxruntime|QnnHtp|QnnSystem).*\\.so")
        private val cacheLock = Any()

        /** Directory holding cached QNN contexts for a given caller cache directory. */
        @JvmStatic fun cacheDirectory(cacheDir: File) = File(cacheDir, CACHE_DIRECTORY)

        private fun requireQnnApi() {
            try {
                OrtSession.SessionOptions::class.java.getMethod("addQnn", Map::class.java)
            } catch (e: NoSuchMethodException) {
                throw UnsupportedOperationException("ONNX Runtime without the QNN execution provider; use onnxruntime-android-qnn", e)
            }
        }

        /** Reflection keeps this file compiling against onnxruntime-android builds without addQnn. */
        private fun addQnn(options: OrtSession.SessionOptions, qnnOptions: Map<String, String>) {
            val method = OrtSession.SessionOptions::class.java.getMethod("addQnn", Map::class.java)
            try {
                method.invoke(options, qnnOptions)
            } catch (e: InvocationTargetException) {
                throw e.targetException ?: e
            }
        }

        /** The HTP stub and skel libraries are found through ADSP_LIBRARY_PATH (';'-separated):
         * the app's extracted native libraries first, then the existing value and vendor DSP paths. */
        private fun configureDspLibraryPath(nativeLibDir: String) {
            require(nativeLibDir.isNotBlank() && File(nativeLibDir).isDirectory) { "Native library directory missing: $nativeLibDir" }
            synchronized(cacheLock) {
                val current = Os.getenv(ADSP_LIBRARY_PATH).orEmpty()
                val paths = LinkedHashSet<String>()
                paths += nativeLibDir
                current.split(';').map { it.trim() }.filterTo(paths) { it.isNotEmpty() }
                paths += VENDOR_DSP_PATHS
                val value = paths.joinToString(";")
                if (value != current) Os.setenv(ADSP_LIBRARY_PATH, value, true)
                Log.i(TAG, "QNN $ADSP_LIBRARY_PATH=$value")
            }
        }

        /** Everything a compiled context depends on. Library contents are hashed, so reinstalling
         * identical binaries keeps the cache while any runtime or OS/device change invalidates it. */
        private fun cacheIdentity(environment: OrtEnvironment, mainFile: File, nativeLibDir: String,
                                  qnnOptions: Map<String, String>): String {
            val libraries = File(nativeLibDir).listFiles { file -> file.isFile && NATIVE_LIBRARY.matches(file.name) }
                .orEmpty().sortedBy { it.name }
            check(libraries.any { it.name == "libQnnHtp.so" }) { "libQnnHtp.so is not in $nativeLibDir; the QNN runtime is not packaged" }
            val lines = ArrayList<String>()
            lines += "format=1"
            lines += "main.sha256=${hex(sha256(mainFile))}"
            lines += "main.bytes=${mainFile.length()}"
            qnnOptions.forEach { (key, value) -> lines += "qnn.$key=$value" }
            lines += "session.disable_cpu_ep_fallback=1"
            lines += "ep.context_embed_mode=1"
            lines += "ort.version=${ortVersion(environment)}"
            libraries.forEach { lines += "lib.${it.name}=${hex(sha256(it))}" }
            lines += "device.manufacturer=${Build.MANUFACTURER}"
            lines += "device.model=${Build.MODEL}"
            lines += "device.hardware=${Build.HARDWARE}"
            lines += "device.board=${Build.BOARD}"
            lines += "device.soc=${Build.SOC_MANUFACTURER}/${Build.SOC_MODEL}"
            lines += "device.fingerprint=${Build.FINGERPRINT}"
            lines += "device.sdk=${Build.VERSION.SDK_INT}"
            return lines.joinToString("\n")
        }

        private fun ortVersion(environment: OrtEnvironment): String = try {
            OrtEnvironment::class.java.getMethod("getVersion").invoke(environment)?.toString() ?: "unknown"
        } catch (e: ReflectiveOperationException) {
            "unknown"
        }

        private fun openSessions(environment: OrtEnvironment, cameraFile: File, mainFile: File, root: File, key: String,
                                 identity: String, qnnOptions: Map<String, String>): Opened {
            var camera: OrtSession? = null
            var main: OrtSession? = null
            try {
                val cameraSession = cameraOptions().use { environment.createSession(cameraFile.absolutePath, it) }
                camera = cameraSession
                val outputs = validateCamera(cameraSession)
                val opened = openMain(environment, mainFile, root, key, identity, qnnOptions) { validateMain(it, cameraSession, outputs) }
                main = opened.session
                // End-to-end check before the caller relies on the session.
                infer(environment, cameraSession, outputs, opened.session, FloatArray(3 * WIDTH * HEIGHT),
                    floatArrayOf(214f, 212f, (WIDTH - 1) / 2f, (HEIGHT - 1) / 2f))
                return Opened(cameraSession, outputs, opened)
            } catch (e: Throwable) {
                closeQuietly(main, e); closeQuietly(camera, e)
                throw e
            }
        }

        private fun openMain(environment: OrtEnvironment, mainFile: File, root: File, key: String, identity: String,
                             qnnOptions: Map<String, String>, validate: (OrtSession) -> Unit): MainSession = synchronized(cacheLock) {
            check(root.isDirectory || root.mkdirs()) { "Cannot create QNN cache directory $root" }
            FileChannel.open(File(root, LOCK_FILE).toPath(), StandardOpenOption.CREATE, StandardOpenOption.WRITE).use { channel ->
                channel.lock().use {
                    removeStaleBuilds(root)
                    val entry = File(root, key)
                    var publish = true
                    if (entry.exists()) {
                        val cached = loadCached(environment, entry, identity, qnnOptions, validate)
                        if (cached != null) return@synchronized MainSession(cached, true, 0)
                        publish = quarantine(root, entry)
                    }
                    build(environment, mainFile, root, entry, key, identity, qnnOptions, validate, publish)
                }
            }
        }

        /** Null when the entry is incomplete, stale, corrupt or fails to load; the caller quarantines it. */
        private fun loadCached(environment: OrtEnvironment, entry: File, identity: String, qnnOptions: Map<String, String>,
                               validate: (OrtSession) -> Unit): OrtSession? {
            val context = File(entry, CONTEXT_FILE)
            val problem = cacheProblem(entry, context, identity)
            if (problem != null) {
                Log.w(TAG, "QNN cache ${entry.name} rejected: $problem")
                return null
            }
            // Provider setup errors are not cache errors: let them propagate without quarantining.
            val options = mainOptions(qnnOptions, null)
            val started = SystemClock.elapsedRealtime()
            var session: OrtSession? = null
            return try {
                session = options.use { environment.createSession(context.absolutePath, it) }
                validate(session)
                Log.i(TAG, "QNN context ${entry.name} loaded in ${SystemClock.elapsedRealtime() - started} ms")
                session
            } catch (e: Exception) {
                closeQuietly(session, e)
                // Checksums and runtime identity already matched. A transient
                // DSP failure must fall back promptly, not trigger minutes of
                // recompilation of an intact context.
                Log.w(TAG, "QNN cache ${entry.name} failed to load; retaining intact context", e)
                throw e
            }
        }

        private fun cacheProblem(entry: File, context: File, identity: String): String? = try {
            val manifestFile = File(entry, MANIFEST_FILE)
            val manifest = Properties()
            when {
                !manifestFile.isFile -> "manifest missing"
                !context.isFile || context.length() <= 0 -> "context missing"
                else -> {
                    FileInputStream(manifestFile).use { manifest.load(it) }
                    when {
                        manifest.getProperty("format") != "1" -> "unknown manifest format"
                        manifest.getProperty("identity.sha256") != hex(sha256(identity.toByteArray(Charsets.UTF_8))) -> "identity changed"
                        manifest.getProperty("context.bytes") != context.length().toString() -> "context size changed"
                        manifest.getProperty("context.sha256") != hex(sha256(context)) -> "context checksum mismatch"
                        else -> null
                    }
                }
            }
        } catch (e: Exception) {
            "manifest unreadable: ${e.message}"
        }

        /** Compiles into a private staging directory, then publishes the whole directory with one rename. */
        private fun build(environment: OrtEnvironment, mainFile: File, root: File, entry: File, key: String, identity: String,
                          qnnOptions: Map<String, String>, validate: (OrtSession) -> Unit, publish: Boolean): MainSession {
            val staging = File(root, ".build-$key-${UUID.randomUUID()}")
            val context = File(staging, CONTEXT_FILE)
            val options = mainOptions(qnnOptions, context)
            if (!staging.mkdir()) {
                options.close()
                throw IllegalStateException("Cannot create QNN staging directory $staging")
            }
            val started = SystemClock.elapsedRealtime()
            var compiled: OrtSession? = null
            try {
                Log.i(TAG, "QNN compiling ${mainFile.name} for cache entry $key; this takes minutes on first use")
                compiled = options.use { environment.createSession(mainFile.absolutePath, it) }
                validate(compiled)
                val compileMillis = SystemClock.elapsedRealtime() - started
                check(context.isFile && context.length() > 0) { "QNN did not write a context model to $context" }
                if (!publish) {
                    Log.w(TAG, "QNN context compiled in $compileMillis ms but not published: entry $key could not be quarantined")
                    deleteOwnedTree(staging)
                    return MainSession(compiled, false, compileMillis)
                }
                syncFile(context)
                writeManifest(staging, identity, context, compileMillis)
                Files.move(staging.toPath(), entry.toPath(), StandardCopyOption.ATOMIC_MOVE)
                Log.i(TAG, "QNN context compiled in $compileMillis ms and published as $entry")
                // Release the expensive compilation graph before opening the
                // cached graph; two full NPU sessions need not coexist.
                val superseded = checkNotNull(compiled)
                compiled = null
                try { superseded.close() } catch (e: Exception) { Log.w(TAG, "Closing the QNN compile session failed", e) }
                val reloaded = checkNotNull(loadCached(environment, entry, identity, qnnOptions, validate)) {
                    "Published QNN context failed integrity validation"
                }
                return MainSession(reloaded, false, compileMillis)
            } catch (e: Throwable) {
                closeQuietly(compiled, e)
                if (staging.exists()) deleteOwnedTree(staging)
                throw e
            }
        }

        private fun writeManifest(directory: File, identity: String, context: File, compileMillis: Long) {
            val manifest = Properties()
            manifest.setProperty("format", "1")
            manifest.setProperty("identity.sha256", hex(sha256(identity.toByteArray(Charsets.UTF_8))))
            manifest.setProperty("identity", identity)
            manifest.setProperty("context.file", context.name)
            manifest.setProperty("context.bytes", context.length().toString())
            manifest.setProperty("context.sha256", hex(sha256(context)))
            manifest.setProperty("compile.millis", compileMillis.toString())
            manifest.setProperty("created.epoch.millis", System.currentTimeMillis().toString())
            val temporary = File(directory, "$MANIFEST_FILE.tmp")
            FileOutputStream(temporary).use { output ->
                manifest.store(output, "QnnDepthSession cache entry")
                output.fd.sync()
            }
            Files.move(temporary.toPath(), File(directory, MANIFEST_FILE).toPath(), StandardCopyOption.ATOMIC_MOVE)
        }

        /** Moves an entry aside instead of deleting it; keeps only the newest [MAX_QUARANTINED]. */
        private fun quarantine(root: File, entry: File): Boolean {
            val directory = File(root, QUARANTINE_DIRECTORY)
            if (!directory.isDirectory && !directory.mkdirs()) {
                Log.w(TAG, "Cannot create $directory; QNN cache entry ${entry.name} left in place")
                return false
            }
            val target = File(directory, "${entry.name}-${System.currentTimeMillis()}")
            return try {
                Files.move(entry.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
                Log.w(TAG, "QNN cache entry ${entry.name} quarantined to $target")
                directory.listFiles { file -> file.isDirectory && QUARANTINE_NAME.matches(file.name) }.orEmpty()
                    .sortedByDescending { it.name.substringAfterLast('-').toLong() }
                    .drop(MAX_QUARANTINED).forEach { deleteOwnedTree(it) }
                true
            } catch (e: IOException) {
                Log.w(TAG, "Cannot quarantine QNN cache entry ${entry.name}", e)
                false
            }
        }

        /** Staging directories left by a crashed build; safe to remove while holding the cache lock. */
        private fun removeStaleBuilds(root: File) {
            root.listFiles { file -> file.isDirectory && BUILD_NAME.matches(file.name) }.orEmpty().forEach {
                Log.w(TAG, "Removing interrupted QNN build ${it.name}")
                deleteOwnedTree(it)
            }
        }

        /** Deletes a directory this class created. Symbolic links are removed, never followed. */
        private fun deleteOwnedTree(directory: File) {
            try {
                Files.walkFileTree(directory.toPath(), object : SimpleFileVisitor<Path>() {
                    override fun visitFile(file: Path, attrs: BasicFileAttributes): FileVisitResult {
                        Files.delete(file); return FileVisitResult.CONTINUE
                    }
                    override fun postVisitDirectory(dir: Path, exc: IOException?): FileVisitResult {
                        if (exc != null) throw exc
                        Files.delete(dir); return FileVisitResult.CONTINUE
                    }
                })
            } catch (e: IOException) {
                Log.w(TAG, "Cannot remove $directory", e)
            }
        }

        private fun cameraOptions(): OrtSession.SessionOptions {
            val options = OrtSession.SessionOptions()
            try {
                options.setIntraOpNumThreads(1)
                options.setInterOpNumThreads(1)
                options.addConfigEntry("session.intra_op.allow_spinning", "0")
                options.addConfigEntry("session.inter_op.allow_spinning", "0")
                return options
            } catch (e: Throwable) {
                options.close()
                throw e
            }
        }

        /** [contextOutput] non-null: compile and write the EP context model (embedded binary) there. */
        private fun mainOptions(qnnOptions: Map<String, String>, contextOutput: File?): OrtSession.SessionOptions {
            val options = OrtSession.SessionOptions()
            try {
                options.setIntraOpNumThreads(1)
                options.addConfigEntry("session.intra_op.allow_spinning", "0")
                options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                if (contextOutput != null) {
                    options.addConfigEntry("ep.context_enable", "1")
                    options.addConfigEntry("ep.context_embed_mode", "1")
                    options.addConfigEntry("ep.context_file_path", contextOutput.absolutePath)
                }
                addQnn(options, qnnOptions)
                return options
            } catch (e: Throwable) {
                options.close()
                throw e
            }
        }

        private fun validateCamera(camera: OrtSession): List<String> {
            check(camera.inputNames == setOf(K_INPUT)) { "QNN camera model inputs ${camera.inputNames}, expected [$K_INPUT]" }
            check(shape(camera.inputInfo.getValue(K_INPUT)).contentEquals(longArrayOf(1, 3, 3))) { "QNN camera K input must be [1,3,3]" }
            val outputs = camera.outputNames.toList()
            check(outputs.isNotEmpty()) { "QNN camera model has no outputs" }
            return outputs
        }

        private fun validateMain(main: OrtSession, camera: OrtSession, cameraOutputs: List<String>) {
            check(main.inputNames == setOf(IMAGE_INPUT) + cameraOutputs) {
                "QNN main inputs ${main.inputNames} do not match image + camera outputs $cameraOutputs"
            }
            check(shape(main.inputInfo.getValue(IMAGE_INPUT)).contentEquals(longArrayOf(1, 3, HEIGHT.toLong(), WIDTH.toLong()))) {
                "QNN main image input must be [1,3,$HEIGHT,$WIDTH]"
            }
            for (name in cameraOutputs) {
                val produced = shape(camera.outputInfo.getValue(name))
                val consumed = shape(main.inputInfo.getValue(name))
                check(produced.all { it > 0 } && produced.contentEquals(consumed)) {
                    "QNN camera tensor $name is ${produced.contentToString()} but the main graph takes ${consumed.contentToString()}"
                }
            }
            check(main.outputNames.contains(DEPTH_OUTPUT)) { "QNN main has no $DEPTH_OUTPUT output" }
            check(shape(main.outputInfo.getValue(DEPTH_OUTPUT)).contentEquals(longArrayOf(1, HEIGHT.toLong(), WIDTH.toLong()))) {
                "QNN depth output must be [1,$HEIGHT,$WIDTH]"
            }
        }

        private fun shape(info: NodeInfo): LongArray = (info.info as? TensorInfo)?.shape ?: error("${info.name} is not a tensor")

        private fun infer(environment: OrtEnvironment, camera: OrtSession, cameraOutputs: List<String>, main: OrtSession,
                          chw: FloatArray, k: FloatArray): FloatArray {
            require(chw.size == 3 * WIDTH * HEIGHT) { "QNN depth input must be 3x${HEIGHT}x$WIDTH values, got ${chw.size}" }
            require(k.size == 4 && k.all { it.isFinite() } && k[0] > 0f && k[1] > 0f) { "Intrinsics must be finite fx,fy,cx,cy with positive focal lengths" }
            for (value in chw) require(value.isFinite()) { "QNN depth input contains a non-finite value" }
            val matrix = floatArrayOf(k[0], 0f, k[2], 0f, k[1], k[3], 0f, 0f, 1f)
            return OnnxTensor.createTensor(environment, FloatBuffer.wrap(matrix), longArrayOf(1, 3, 3)).use { kTensor ->
                camera.run(mapOf(K_INPUT to kTensor)).use { cameraResult ->
                    OnnxTensor.createTensor(environment, FloatBuffer.wrap(chw), longArrayOf(1, 3, HEIGHT.toLong(), WIDTH.toLong())).use { image ->
                        val feeds = HashMap<String, OnnxTensor>(cameraOutputs.size + 1)
                        feeds[IMAGE_INPUT] = image
                        for (name in cameraOutputs) {
                            feeds[name] = cameraResult.get(name).orElse(null) as? OnnxTensor ?: error("QNN camera output $name missing")
                        }
                        main.run(feeds).use { result ->
                            val depth = result.get(DEPTH_OUTPUT).orElse(null) as? OnnxTensor ?: error("QNN depth output missing")
                            check(depth.info.shape.contentEquals(longArrayOf(1, HEIGHT.toLong(), WIDTH.toLong()))) {
                                "Unexpected QNN depth shape ${depth.info.shape.contentToString()}"
                            }
                            val buffer = depth.floatBuffer
                            val output = FloatArray(WIDTH * HEIGHT)
                            check(buffer.remaining() == output.size) { "QNN depth buffer holds ${buffer.remaining()} values" }
                            buffer.get(output)
                            val invalid = output.count { !it.isFinite() || it <= 0f }
                            check(invalid == 0) { "QNN depth contains $invalid invalid values" }
                            output
                        }
                    }
                }
            }
        }

        private fun syncFile(file: File) = RandomAccessFile(file, "rw").use { it.fd.sync() }

        private fun closeQuietly(session: OrtSession?, cause: Throwable) {
            try { session?.close() } catch (e: Exception) { cause.addSuppressed(e) }
        }

        private fun sha256(bytes: ByteArray): ByteArray = MessageDigest.getInstance("SHA-256").digest(bytes)

        private fun sha256(file: File): ByteArray {
            val digest = MessageDigest.getInstance("SHA-256")
            FileInputStream(file).use { input ->
                val buffer = ByteArray(1 shl 20)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest()
        }

        private fun hex(bytes: ByteArray) = bytes.joinToString("") { "%02x".format(it) }
    }
}
