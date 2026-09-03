plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.serialization")
}

android {
    namespace = "com.recon.poselogger"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.recon.poselogger"
        // RGB-only Camera2 scanner; no ARCore-capable device is required.
        minSdk = 33
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    testOptions { unitTests.isReturnDefaultValues = true }

    // The historical AR prototype remains in the repository for reference,
    // but this application variant ships only the Camera2 RGB scanner/viewer.
    sourceSets {
        getByName("main").java.setSrcDirs(
            listOf("src/main/java/com/recon/poselogger/rgbscan")
        )
        getByName("main").res.setSrcDirs(listOf("src/rgbscan/res"))
        getByName("main").jniLibs.setSrcDirs(emptyList<String>())
        getByName("test").java.setSrcDirs(
            listOf("src/test/java/com/recon/poselogger/rgbscan")
        )
        getByName("androidTest").java.setSrcDirs(
            listOf("src/androidTest/java/com/recon/poselogger/rgbscan")
        )
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
