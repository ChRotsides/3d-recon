package com.recon.poselogger.rgbscan

import android.os.Bundle
import android.graphics.Color
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class CloudViewerActivity : AppCompatActivity() {
    private var cloudView: PointCloudView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            val metadata = File(requireNotNull(intent.getStringExtra(EXTRA_METADATA)))
            val binary = File(requireNotNull(intent.getStringExtra(EXTRA_BINARY)))
            val viewer = PointCloudView(this, PointCloudDecoder.decode(metadata, binary))
            cloudView = viewer
            val hint = TextView(this).apply {
                text = "1 finger orbit  •  2 fingers move/pinch  •  Double-tap to fit"
                setTextColor(Color.WHITE)
                setBackgroundColor(0x99000000.toInt())
                setPadding(24, 16, 24, 16)
            }
            setContentView(FrameLayout(this).apply {
                addView(viewer, FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                ))
                addView(hint, FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.TOP,
                ))
                addView(Button(this@CloudViewerActivity).apply {
                    text = "FIT"
                    setOnClickListener { viewer.fitCloud() }
                }, FrameLayout.LayoutParams(
                    (resources.displayMetrics.density * 88).toInt(),
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.BOTTOM or Gravity.END,
                ))
            })
        } catch (error: Exception) {
            setContentView(TextView(this).apply { text = "Cloud rejected: ${error.message}" })
        }
    }

    override fun onResume() {
        super.onResume()
        cloudView?.onResume()
    }

    override fun onPause() {
        cloudView?.onPause()
        super.onPause()
    }

    companion object {
        const val EXTRA_METADATA = "cloud_metadata"
        const val EXTRA_BINARY = "cloud_binary"
    }
}
