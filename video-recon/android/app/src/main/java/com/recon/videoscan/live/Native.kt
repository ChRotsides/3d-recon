package com.recon.videoscan.live

/** JNI surface of libvivescan: OpenVINS poses and the voxel-hash TSDF, all on the phone. */
object Native {
    init { System.loadLibrary("opencv_java4"); System.loadLibrary("livescan") }
    external fun init(configPath: String): Long
    external fun feedImu(h: Long, t: Double, wx: Double, wy: Double, wz: Double, ax: Double, ay: Double, az: Double)
    /** Returns 0 before initialisation, else 1 + number of landmarks written. pose = [px,py,pz,qx,qy,qz,qw] camera->world (Y up). */
    external fun feedImage(h: Long, t: Double, w: Int, hgt: Int, gray: ByteArray, pose: FloatArray, landmarks: FloatArray): Int
    external fun tsdfReset(h: Long, voxel: Float)
    external fun tsdfIntegrate(h: Long, depthMm: ShortArray, w: Int, hgt: Int, k: FloatArray, r: FloatArray, c: FloatArray, rgba: ByteArray?, rw: Int, rh: Int, stride: Int): FloatArray
    external fun tsdfSurface(h: Long, maxPoints: Int): FloatArray
    external fun destroy(h: Long)
}
