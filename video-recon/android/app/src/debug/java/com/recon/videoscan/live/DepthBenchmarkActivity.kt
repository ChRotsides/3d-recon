package com.recon.videoscan.live

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.view.WindowManager
import android.widget.TextView

/** Debug instrumentation host: keeps timing runs in the foreground without opening a camera. */
class DepthBenchmarkActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(TextView(this).apply {
            text = "Checking Live Scan performance…\nLive Scan will reopen after the check."
            textSize = 20f; gravity = Gravity.CENTER; setPadding(32,32,32,32)
        })
    }
}
