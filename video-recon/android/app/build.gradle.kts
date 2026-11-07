plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.recon.videoscan"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.recon.videoscan"
        // MediaRecorder(Context) and the Camera2 keys used here are API 31+.
        minSdk = 31
        targetSdk = 35
        versionCode = 7
        versionName = "0.7-live-npu"
        testInstrumentationRunner = "com.recon.videoscan.live.RecoveryInstrumentation"
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild { cmake {
            arguments += listOf("-DANDROID_STL=c++_shared")
            val headers = providers.gradleProperty("nativeHeaders").orNull
                ?: rootProject.file("../.deps/native-headers").absolutePath
            arguments += listOf("-DTHIRD_PARTY_INCLUDE=$headers/include", "-DEigen3_DIR=$headers/share/eigen3/cmake")
            for (key in listOf("OV_ROOT", "OV_FULL_ROOT", "OPENCV_SDK", "Ceres_DIR"))
                providers.gradleProperty(key).orNull?.let { arguments += "-D$key=$it" }
            targets += listOf("livescan") // Native test executable is built/run explicitly, never packaged.
        } }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    ndkVersion = "28.2.13676358"
    if (providers.gradleProperty("cpuOnly").orNull != "true") {
        // The DSP loader needs an extracted filesystem path for its skeleton.
        packaging { jniLibs.useLegacyPackaging = true }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    testOptions { unitTests.isReturnDefaultValues = true }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    // on-device depth: DA3-Small exported to ONNX (video-recon/vio/export_da3.py)
    // QNN retains the CPU provider for other models and fallback. cpuOnly is a
    // diagnostic build for devices that cannot load the Qualcomm runtime.
    if (providers.gradleProperty("cpuOnly").orNull != "true")
        implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.22.0")
    else implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
