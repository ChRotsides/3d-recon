package com.recon.videoscan.live

/** ONNX depth exports use HxW, 1xHxW or singleton batch/channel axes. */
data class DepthTensorLayout(val width: Int, val height: Int) {
    companion object {
        fun from(shape: LongArray): DepthTensorLayout {
            require(shape.size in 2..5 && shape.dropLast(2).all { it == 1L }) {
                "Expected one depth image, got ${shape.contentToString()}"
            }
            val h = shape[shape.size - 2]; val w = shape.last()
            require(h in 1..4096 && w in 1..4096) { "Invalid depth image dimensions" }
            return DepthTensorLayout(w.toInt(), h.toInt())
        }
    }
}
