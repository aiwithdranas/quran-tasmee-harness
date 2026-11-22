// PoC Harness app module - Stage 1 ONLY. Temporary test app.
// The Quran Lab SDK 4.0 dependency is active (see below) and the real
// gateway file (RealTajweedGateway.kt) is written and wired in
// MainActivity (real gateway when a key is present, STUB otherwise).
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "org.quranapp.poc.harness"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.quranapp.poc.harness"
        minSdk = 26 // Quran Lab Android SDK minimum (AAR, arm64-v8a + armeabi-v7a)
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0-poc"

        // Key injection: QURANLAB_KEY comes ONLY from local.properties
        // (git-ignored). Never hardcode a key here or in any source file.
        // NOTE: project.findProperty does NOT read local.properties, so
        // the file must be parsed explicitly — otherwise the key silently
        // stays empty and the app runs the STUB gateway (no recognition,
        // "take_finished heads=none"). Found 2026-11-21: the first APK
        // shipped without the key for exactly this reason.
        val quranLabKey: String = run {
            val localFile = rootProject.file("local.properties")
            var fromFile: String? = null
            if (localFile.exists()) {
                fromFile = localFile.readLines()
                    .firstOrNull { it.startsWith("QURANLAB_KEY=") }
                    ?.substringAfter("=")
                    ?.trim()
            }
            fromFile
                ?: project.findProperty("QURANLAB_KEY") as String?
                ?: System.getenv("QURANLAB_KEY") ?: ""
        }
        buildConfigField("String", "QURANLAB_KEY", "\"$quranLabKey\"")
        buildConfigField("boolean", "HAS_QURANLAB_KEY", quranLabKey.isNotBlank().toString())
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        buildConfig = true
        compose = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.compose.ui:ui:1.7.0")
    implementation("androidx.compose.material3:material3:1.3.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.5")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.5")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // SDK 4.0 confirmed: ai.quranlab:tajweedformer:4.0.1 (Maven Central,
    // latest as of 2026-11-21, and the official SDK 4.0 Android quick
    // start at sdk.quranlab.ai/docs/4/quickstart/android). SDK 3.x is
    // NOT used: it stops receiving the model on 2027-01-15. The key
    // flow is complete (Development key in local.properties).
    implementation("ai.quranlab:tajweedformer:4.0.1")
}
