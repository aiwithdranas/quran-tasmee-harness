// Quran Tasmee' PoC Harness - TEMPORARY test app ONLY (Stage 1)
// NOT the full Quran app. Single-screen Android test app, installed via
// ADB on the user's phone, never published, never in any store.
// ApplicationId: org.quranapp.poc.harness
// See README.md in this folder before building.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "quran-tasmee-poc-harness"
include(":app")
