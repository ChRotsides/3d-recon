package com.recon.videoscan.live

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.ArrayDeque
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.sqrt

/** Rolling-window v2 learned odometry. Accurate VIO/map poses anchor its world frame;
 * between anchors it publishes provisional motion while relocalization keeps running.
 */
class PoseTransformerRunner(modelFile: File) : AutoCloseable {
    data class Estimate(val pose: FloatArray, val timestamp: Double, val sigmaMeters: Float,
        val features: Int, val inferenceMs: Double)

    private data class ImuSample(val t: Double, val g: DoubleArray, val a: DoubleArray)
    private data class Track(val x: Float, val y: Float, val age: Float)
    private data class PackedImu(val values: FloatArray, val mask: ByteArray, val rotation: FloatArray)
    private data class Frame(val epoch: Long, val timestamp: Double, val features: FloatArray,
        val featureMask: ByteArray, val featureCount: Int, val imu: FloatArray, val imuMask: ByteArray,
        val calibration: FloatArray, val dt: Float, val imuRotation: FloatArray, val anchor: FloatArray?)

    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "pose-transformer").apply { isDaemon = true }
    }
    private val workerRunning = AtomicBoolean(false)
    private val inputLock = Any()
    private val imuSamples = ArrayDeque<ImuSample>()
    private val pending = ArrayDeque<Frame>()
    private var previousTracks = emptyMap<Long, Track>()
    private var previousCameraTime: Double? = null
    private var previousOffset = 0.0
    private var cachedCalibration: DoubleArray? = null
    private var epoch = 0L
    @Volatile private var closed = false

    // Worker-thread state.
    private val history = ArrayDeque<Frame>()
    private var modelEpoch = -1L
    private var position = FloatArray(3)
    private var rotationCv = identity()
    private var anchored = false
    private var inferenceCount = 0L

    @Volatile private var latest: Estimate? = null
    @Volatile var state: String = "warming up"
        private set

    init {
        require(modelFile.isFile) { "missing pose transformer: $modelFile" }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = environment.createSession(modelFile.absolutePath, options)
        require(session.inputNames == setOf("features", "feature_mask", "imu", "imu_mask", "calibration", "dt", "imu_rotation"))
        require(session.outputNames.containsAll(setOf("delta_position", "delta_rotation", "log_variance")))
    }

    fun estimate(): Estimate? = latest?.let { it.copy(pose = it.pose.copyOf()) }

    fun addImu(sample: VioInputQueue.Imu) = synchronized(inputLock) {
        if (closed) return@synchronized
        val a = sample.acceleration; val g = sample.gyro
        imuSamples.addLast(ImuSample(a.t, doubleArrayOf(g.x, g.y, g.z), doubleArrayOf(a.x, a.y, a.z)))
        while (imuSamples.size > 1024) imuSamples.removeFirst()
    }

    /** The native feature snapshot is count/header followed by id/xn/yn triples. */
    fun offer(timestamp: Double, gray: ByteArray, width: Int, height: Int,
        nativeFeatures: DoubleArray, calibration: DoubleArray, cameraImuOffset: Double,
        accuratePose: FloatArray?) = synchronized(inputLock) {
        if (closed || !timestamp.isFinite() || nativeFeatures.size < 14) return@synchronized
        if (calibration.size == 11 && calibration.all { it.isFinite() } && calibration[0] > 0.0)
            cachedCalibration = calibration.copyOf()
        val camera = cachedCalibration ?: return@synchronized
        val count = nativeFeatures[0].toInt()
        if (count !in 0..MAX_FEATURES || nativeFeatures.size != 14 + 3 * count) return@synchronized
        val current = LinkedHashMap<Long, Track>(count)
        repeat(count) { i ->
            val id = nativeFeatures[14 + 3 * i].toLong()
            val x = nativeFeatures[15 + 3 * i].toFloat(); val y = nativeFeatures[16 + 3 * i].toFloat()
            if (x.isFinite() && y.isFinite()) current[id] = Track(x, y, 0f)
        }
        val oldTime = previousCameraTime
        if (oldTime == null) {
            previousCameraTime = timestamp; previousOffset = cameraImuOffset; previousTracks = current
            return@synchronized
        }
        val dt = timestamp - oldTime
        if (!dt.isFinite() || dt <= 0.0 || dt > 1.5) {
            previousCameraTime = timestamp; previousOffset = cameraImuOffset; previousTracks = current
            state = "waiting for a continuous camera interval"
            return@synchronized
        }
        val rotation = FloatArray(9) { nativeFeatures[5 + it].toFloat() }
        if (rotation.any { !it.isFinite() }) return@synchronized
        val packed = packImu(oldTime + previousOffset, timestamp + cameraImuOffset, rotation)
        val features = FloatArray(MAX_FEATURES * FEATURE_DIM)
        val mask = ByteArray(MAX_FEATURES)
        var used = 0
        for ((id, now) in current) {
            val before = previousTracks[id] ?: continue
            if (used == MAX_FEATURES) break
            val age = (before.age + dt).coerceAtMost(5.0).toFloat()
            val patch = patchStats(gray, width, height, camera, now.x, now.y)
            val o = used * FEATURE_DIM
            features[o] = before.x; features[o + 1] = before.y
            features[o + 2] = now.x; features[o + 3] = now.y
            features[o + 4] = now.x - before.x; features[o + 5] = now.y - before.y
            features[o + 6] = age
            features[o + 7] = 0f // OpenVINS has already round-trip/geometrically filtered this track.
            features[o + 8] = 0f
            features[o + 9] = patch[0]; features[o + 10] = patch[1]
            features[o + 11] = dt.toFloat()
            mask[used++] = 1
            current[id] = now.copy(age = age)
        }
        previousCameraTime = timestamp; previousOffset = cameraImuOffset; previousTracks = current
        if (packed == null) {
            state = "waiting for IMU coverage"
            return@synchronized
        }
        val c = FloatArray(20)
        c[0] = (camera[0] / camera[8]).toFloat(); c[1] = (camera[1] / camera[9]).toFloat()
        c[2] = (camera[2] / camera[8]).toFloat(); c[3] = (camera[3] / camera[9]).toFloat()
        for (i in 0 until 9) c[4 + i] = rotation[i]
        // The phone camera/IMU lever arm is only centimetres and was randomized during training.
        c[16] = (camera[8] / camera[9]).toFloat(); c[17] = cameraImuOffset.toFloat()
        c[18] = 1.7e-3f; c[19] = 2.0e-2f
        pending.addLast(Frame(epoch, timestamp, features, mask, used, packed.values, packed.mask,
            c, dt.toFloat(), packed.rotation, accuratePose?.copyOf()))
        while (imuSamples.size > 2 && imuSamples.elementAt(1).t <= oldTime + previousOffset) imuSamples.removeFirst()
        scheduleWorker()
    }

    /** Keep inertial/model continuity through a VIO restart but do not join reused native track IDs. */
    fun breakVisualTracks() = synchronized(inputLock) { previousTracks = emptyMap() }

    fun reset() = synchronized(inputLock) {
        epoch++
        pending.clear(); imuSamples.clear(); previousTracks = emptyMap(); previousCameraTime = null
        cachedCalibration = null; latest = null; state = "warming up"
    }

    private fun scheduleWorker() {
        if (!workerRunning.compareAndSet(false, true)) return
        executor.execute {
            try {
                while (!closed) {
                    val batch = synchronized(inputLock) {
                        if (pending.isEmpty()) emptyList() else ArrayList<Frame>(pending.size).also {
                            while (pending.isNotEmpty()) it.add(pending.removeFirst())
                        }
                    }
                    if (batch.isEmpty()) break
                    infer(batch)
                }
            } catch (error: Throwable) {
                state = "model error: ${error.message}"
            } finally {
                workerRunning.set(false)
                val again = synchronized(inputLock) { pending.isNotEmpty() && !closed }
                if (again) scheduleWorker()
            }
        }
    }

    private fun infer(newFrames: List<Frame>) {
        if (newFrames.last().epoch != modelEpoch) {
            history.clear(); position = FloatArray(3); rotationCv = identity(); anchored = false
            modelEpoch = newFrames.last().epoch
        }
        for (frame in newFrames.filter { it.epoch == modelEpoch }) history.addLast(frame)
        while (history.size > CONTEXT) history.removeFirst()
        val frames = history.toList(); val n = frames.size
        val features = FloatArray(n * MAX_FEATURES * FEATURE_DIM)
        val featureMask = ByteArray(n * MAX_FEATURES)
        val imu = FloatArray(n * MAX_IMU * 7); val imuMask = ByteArray(n * MAX_IMU)
        val calibration = FloatArray(n * 20); val dt = FloatArray(n); val imuRotation = FloatArray(n * 9)
        frames.forEachIndexed { i, frame ->
            frame.features.copyInto(features, i * MAX_FEATURES * FEATURE_DIM)
            frame.featureMask.copyInto(featureMask, i * MAX_FEATURES)
            frame.imu.copyInto(imu, i * MAX_IMU * 7); frame.imuMask.copyInto(imuMask, i * MAX_IMU)
            frame.calibration.copyInto(calibration, i * 20); dt[i] = frame.dt
            frame.imuRotation.copyInto(imuRotation, i * 9)
        }
        val tensors = linkedMapOf<String, OnnxTensor>()
        try {
            tensors["features"] = floatTensor(features, longArrayOf(1, n.toLong(), MAX_FEATURES.toLong(), FEATURE_DIM.toLong()))
            tensors["feature_mask"] = boolTensor(featureMask, longArrayOf(1, n.toLong(), MAX_FEATURES.toLong()))
            tensors["imu"] = floatTensor(imu, longArrayOf(1, n.toLong(), MAX_IMU.toLong(), 7))
            tensors["imu_mask"] = boolTensor(imuMask, longArrayOf(1, n.toLong(), MAX_IMU.toLong()))
            tensors["calibration"] = floatTensor(calibration, longArrayOf(1, n.toLong(), 20))
            tensors["dt"] = floatTensor(dt, longArrayOf(1, n.toLong()))
            tensors["imu_rotation"] = floatTensor(imuRotation, longArrayOf(1, n.toLong(), 3, 3))
            val started = System.nanoTime()
            session.run(tensors).use { result ->
                val elapsedMs = (System.nanoTime() - started) / 1e6
                val dp = (result.get("delta_position").orElseThrow() as OnnxTensor).floatBuffer
                val dr = (result.get("delta_rotation").orElseThrow() as OnnxTensor).floatBuffer
                val variance = (result.get("log_variance").orElseThrow() as OnnxTensor).floatBuffer
                val accepted = newFrames.takeLast(minOf(newFrames.size, n))
                val start = n - accepted.size
                accepted.forEachIndexed { j, frame ->
                    val index = start + j
                    if (frame.anchor != null) anchor(frame.anchor)
                    else if (anchored) integrate(dp, dr, index)
                }
                val last = accepted.lastOrNull() ?: return@use
                if (!anchored) {
                    // The session origin and the model origin are both identity until the first VIO anchor.
                    anchored = true
                }
                var sigma2 = 0.0
                for (axis in 0..2) sigma2 += kotlin.math.exp(variance.get((n - 1) * 6 + axis).toDouble())
                if (synchronized(inputLock) { last.epoch == epoch && !closed }) {
                    latest = Estimate(pose(), last.timestamp, (sqrt(sigma2) * last.dt).toFloat(),
                        last.featureCount, elapsedMs)
                    state = "running"
                    inferenceCount++
                    if (inferenceCount == 1L || inferenceCount % 30L == 0L)
                        Log.i("livescan", "pose transformer run=$inferenceCount window=$n pending=${accepted.size} features=${last.featureCount} inference=${"%.1f".format(elapsedMs)}ms")
                }
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    private fun integrate(dp: FloatBuffer, dr: FloatBuffer, index: Int) {
        val delta = FloatArray(3) { dp.get(index * 3 + it) }
        val deltaR = FloatArray(9) { dr.get(index * 9 + it) }
        if (delta.any { !it.isFinite() } || deltaR.any { !it.isFinite() }) return
        val distance = sqrt(delta.sumOf { (it * it).toDouble() })
        val angle = acos(((deltaR[0] + deltaR[4] + deltaR[8] - 1f) * .5f).coerceIn(-1f, 1f).toDouble())
        if (distance > .75 || angle > 1.0) return
        val world = matVec(rotationCv, delta)
        for (i in 0..2) position[i] += world[i]
        rotationCv = orthonormalize(matMul(rotationCv, deltaR))
    }

    private fun anchor(pose: FloatArray) {
        if (pose.size != 7 || pose.any { !it.isFinite() }) return
        position = pose.copyOfRange(0, 3)
        val gl = quaternionMatrix(pose[3], pose[4], pose[5], pose[6])
        rotationCv = gl.copyOf().also { for (row in 0..2) { it[3 * row + 1] *= -1; it[3 * row + 2] *= -1 } }
        anchored = true
    }

    private fun pose(): FloatArray {
        val gl = rotationCv.copyOf().also { for (row in 0..2) { it[3 * row + 1] *= -1; it[3 * row + 2] *= -1 } }
        val q = matrixQuaternion(gl)
        return floatArrayOf(position[0], position[1], position[2], q[0], q[1], q[2], q[3])
    }

    private fun packImu(start: Double, end: Double, rBodyCamera: FloatArray): PackedImu? {
        if (end <= start || imuSamples.size < 2 || imuSamples.first.t > start || imuSamples.last.t < end) return null
        var index = 0
        while (index + 1 < imuSamples.size && imuSamples.elementAt(index + 1).t <= start) index++
        data class Segment(val g: DoubleArray, val a: DoubleArray, val duration: Double)
        val raw = ArrayList<Segment>(); var cursor = start
        while (cursor < end - 1e-9 && index + 1 < imuSamples.size) {
            val sample = imuSamples.elementAt(index); val stop = minOf(end, imuSamples.elementAt(index + 1).t)
            if (stop > cursor) raw.add(Segment(sample.g, sample.a, stop - cursor))
            cursor = stop; index++
        }
        if (cursor < end - 1e-7 || raw.isEmpty()) return null
        val segments = if (raw.size <= MAX_IMU) raw else (0 until MAX_IMU).map { group ->
            val from = group * raw.size / MAX_IMU; val to = (group + 1) * raw.size / MAX_IMU
            val duration = raw.subList(from, to).sumOf { it.duration }
            val g = DoubleArray(3); val a = DoubleArray(3)
            for (item in raw.subList(from, to)) for (axis in 0..2) {
                g[axis] += item.g[axis] * item.duration / duration
                a[axis] += item.a[axis] * item.duration / duration
            }
            Segment(g, a, duration)
        }
        val values = FloatArray(MAX_IMU * 7); val mask = ByteArray(MAX_IMU)
        var r = identity()
        segments.forEachIndexed { i, item ->
            for (axis in 0..2) { values[7 * i + axis] = item.g[axis].toFloat(); values[7 * i + 3 + axis] = item.a[axis].toFloat() }
            values[7 * i + 6] = item.duration.toFloat(); mask[i] = 1
            r = matMul(r, rodrigues(FloatArray(3) { (item.g[it] * item.duration).toFloat() }))
        }
        val cameraRotation = matMul(matMul(transpose(rBodyCamera), r), rBodyCamera)
        return PackedImu(values, mask, cameraRotation)
    }

    private fun patchStats(gray: ByteArray, width: Int, height: Int, c: DoubleArray, xn: Float, yn: Float): FloatArray {
        val r2 = xn * xn + yn * yn
        val xd: Double; val yd: Double
        if (c[10] == 0.0) {
            val radial = 1 + c[4] * r2 + c[5] * r2 * r2
            xd = xn * radial + 2 * c[6] * xn * yn + c[7] * (r2 + 2 * xn * xn)
            yd = yn * radial + c[6] * (r2 + 2 * yn * yn) + 2 * c[7] * xn * yn
        } else {
            val radius = sqrt(r2.toDouble()); val theta = kotlin.math.atan(radius); val t2 = theta * theta
            val scale = if (radius < 1e-12) 1.0 else theta * (1 + t2 * (c[4] + t2 * (c[5] + t2 * (c[6] + t2 * c[7])))) / radius
            xd = xn * scale; yd = yn * scale
        }
        val x = floor(c[0] * xd + c[2] + .5).toInt(); val y = floor(c[1] * yd + c[3] + .5).toInt()
        if (x !in 2 until width - 2 || y !in 2 until height - 2) return floatArrayOf(0f, 0f)
        var sum = 0.0; var squares = 0.0
        for (yy in y - 2..y + 2) for (xx in x - 2..x + 2) {
            val value = (gray[yy * width + xx].toInt() and 255) / 255.0
            sum += value; squares += value * value
        }
        val mean = sum / 25
        return floatArrayOf(mean.toFloat(), sqrt(maxOf(0.0, squares / 25 - mean * mean)).toFloat())
    }

    private fun floatTensor(values: FloatArray, shape: LongArray) =
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), shape)

    private fun boolTensor(values: ByteArray, shape: LongArray): OnnxTensor {
        val buffer = ByteBuffer.allocateDirect(values.size).order(ByteOrder.nativeOrder())
        buffer.put(values).flip()
        return OnnxTensor.createTensor(environment, buffer, shape, OnnxJavaType.BOOL)
    }

    override fun close() {
        synchronized(inputLock) { closed = true; pending.clear() }
        executor.shutdown()
        executor.awaitTermination(1, TimeUnit.SECONDS)
        session.close()
    }

    companion object {
        const val MODEL_FILE = "pose_transformer_v2.onnx"
        const val CONTEXT = 96
        const val MAX_FEATURES = 192
        const val MAX_IMU = 32
        const val FEATURE_DIM = 12

        private fun identity() = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        private fun transpose(a: FloatArray) = FloatArray(9) { i -> a[(i % 3) * 3 + i / 3] }
        private fun matMul(a: FloatArray, b: FloatArray) = FloatArray(9) { i ->
            val row = i / 3; val col = i % 3
            a[3 * row] * b[col] + a[3 * row + 1] * b[3 + col] + a[3 * row + 2] * b[6 + col]
        }
        private fun matVec(a: FloatArray, b: FloatArray) = FloatArray(3) { row ->
            a[3 * row] * b[0] + a[3 * row + 1] * b[1] + a[3 * row + 2] * b[2]
        }
        private fun rodrigues(v: FloatArray): FloatArray {
            val theta = sqrt(v.sumOf { (it * it).toDouble() }).toFloat()
            if (theta < 1e-7f) return floatArrayOf(1f, -v[2], v[1], v[2], 1f, -v[0], -v[1], v[0], 1f)
            val x = v[0] / theta; val y = v[1] / theta; val z = v[2] / theta
            val c = cos(theta); val s = sin(theta); val d = 1 - c
            return floatArrayOf(c + x*x*d, x*y*d - z*s, x*z*d + y*s,
                y*x*d + z*s, c + y*y*d, y*z*d - x*s,
                z*x*d - y*s, z*y*d + x*s, c + z*z*d)
        }
        private fun quaternionMatrix(x: Float, y: Float, z: Float, w: Float): FloatArray {
            val n = sqrt((x*x + y*y + z*z + w*w).toDouble()).toFloat().coerceAtLeast(1e-8f)
            val qx=x/n; val qy=y/n; val qz=z/n; val qw=w/n
            return floatArrayOf(1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw),
                2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw),
                2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy))
        }
        private fun matrixQuaternion(m: FloatArray): FloatArray {
            val trace = m[0] + m[4] + m[8]
            val q = FloatArray(4)
            if (trace > 0f) {
                val s = 2f * sqrt((trace + 1).toDouble()).toFloat(); q[3] = .25f*s
                q[0]=(m[7]-m[5])/s; q[1]=(m[2]-m[6])/s; q[2]=(m[3]-m[1])/s
            } else {
                val i = if (m[0] > m[4] && m[0] > m[8]) 0 else if (m[4] > m[8]) 1 else 2
                if (i == 0) { val s=2f*sqrt((1+m[0]-m[4]-m[8]).toDouble()).toFloat();q[3]=(m[7]-m[5])/s;q[0]=.25f*s;q[1]=(m[1]+m[3])/s;q[2]=(m[2]+m[6])/s }
                else if (i == 1) { val s=2f*sqrt((1+m[4]-m[0]-m[8]).toDouble()).toFloat();q[3]=(m[2]-m[6])/s;q[0]=(m[1]+m[3])/s;q[1]=.25f*s;q[2]=(m[5]+m[7])/s }
                else { val s=2f*sqrt((1+m[8]-m[0]-m[4]).toDouble()).toFloat();q[3]=(m[3]-m[1])/s;q[0]=(m[2]+m[6])/s;q[1]=(m[5]+m[7])/s;q[2]=.25f*s }
            }
            return q
        }
        private fun orthonormalize(m: FloatArray): FloatArray {
            fun norm(v: FloatArray): FloatArray { val n=sqrt(v.sumOf { (it*it).toDouble() }).toFloat().coerceAtLeast(1e-8f); return FloatArray(3){v[it]/n} }
            val x=norm(floatArrayOf(m[0],m[3],m[6]));val y0=floatArrayOf(m[1],m[4],m[7]);val dot=x.indices.sumOf{(x[it]*y0[it]).toDouble()}.toFloat();val y=norm(FloatArray(3){y0[it]-dot*x[it]})
            val z=floatArrayOf(x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0])
            return floatArrayOf(x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2])
        }
    }
}
