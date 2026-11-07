package com.recon.videoscan.live

/** JNI surface of libvivescan: OpenVINS poses and the voxel-hash TSDF, all on the phone. */
object Native {
    init { System.loadLibrary("opencv_java4"); System.loadLibrary("livescan") }
    external fun init(configPath: String, biasPrior: DoubleArray): Long
    external fun biasPrior(h: Long): DoubleArray
    external fun retainMap(old: Long, next: Long)
    external fun mapFrame(h: Long, t: Double, reliable: Boolean, valid: Boolean, pose: FloatArray, landmarks: FloatArray, count: Int, candidates: FloatArray?, descriptors: FloatArray?, invalidateAlignment: Boolean = false, calibration: DoubleArray? = null): DoubleArray
    external fun archiveSurface(h: Long, path: String): Long
    external fun feedImu(h: Long, t: Double, wx: Double, wy: Double, wz: Double, ax: Double, ay: Double, az: Double)
    /** t_imu = t_camera + offset, including current online calibration. */
    external fun cameraImuOffset(h: Long): Double
    /** Returns <= 0 unless state time matches t, else 1 + number of landmarks written.
     * pose = [px,py,pz,qx,qy,qz,qw] camera->world (Y up, OpenGL camera axes). */
    external fun feedImage(h: Long, t: Double, w: Int, hgt: Int, gray: ByteArray, pose: FloatArray, landmarks: FloatArray, calibration: DoubleArray, diagnostics: DoubleArray): Int
    /** Rank-ordered x/y/score triples for this exact camera frame. */
    external fun submitXFeat(h: Long, t: Double, w: Int, hgt: Int, candidates: FloatArray): Boolean
    /** Disable learned replenishment and restore the native detector fallback. */
    external fun setXFeatEnabled(h: Long, enabled: Boolean)
    /** Snapshot from the latest feedImage: up to 192 tracks; count, offset, gyro bias,
     * row-major R_ItoC, then id/xn/yn triples. */
    external fun featureSamples(h: Long): DoubleArray
    external fun tsdfReset(h: Long, voxel: Float)
    external fun tsdfIntegrate(h: Long, depthMm: ShortArray, w: Int, hgt: Int, k: FloatArray, r: FloatArray, c: FloatArray, rgba: ByteArray?, rw: Int, rh: Int, stride: Int, confidence: ByteArray? = null): FloatArray
    external fun tsdfSurface(h: Long, maxPoints: Int): FloatArray
    external fun destroy(h: Long)
}
