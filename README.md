# PoC Harness — README (Stage 1 ONLY, temporary)

## What this is
A single-screen Android test app that proves (or disproves) the tasmee'
engine on the user's Huawei Nova 5T. It is NOT the Quran app:
no Mushaf display, no cards, no full-app database, no Play release.
It is installed directly via ADB and never published.

## What is ready
- `settings.gradle.kts`, `build.gradle.kts`, `app/build.gradle.kts`
  (minSdk 26, compileSdk/targetSdk 34, Kotlin, Compose).
- `AndroidManifest.xml` (RECORD_AUDIO only; `allowBackup=false`).
- `MainActivity.kt` — one screen: expected text, word highlighting,
  Start/Stop, raw event log.
- `TajweedSdkGateway.kt` — interface + **STUB** (emits nothing).
- `RealTajweedGateway.kt` — **real gateway, written 2026-11-21** against
  SDK 4.0.1 (API verified against the installed AAR + Quran Lab docs):
  `TajweedClient(context, key)` from BuildConfig only,
  `client.prefetch()` + `client.model()`, `model.listen()` with
  `endOnSilenceMs = 1500`, per-chunk token ids + confidences,
  whole-take heads via `session.finish()` (`lastFinalHeads`),
  `model.reference` / `model.calibration` accessors for the build step.
  NOT wired into MainActivity yet (see "Wiring" below).
- `Bridge.kt` — expected-unit lookup from the bundled JSON file
  (`assets/reference_67_32.json`, EXISTS — PROVISIONAL, verify
  against `model.reference` before freezing; see item 4 below).
- `Alignment.kt` — C/S/D/H/U/I/R verdicts, verbal accuracy and
  independent recall, with the PoC worked example as a self-check.
- `EventLog.kt` — raw timestamped log; scores are recomputed from it.

## What is NOT ready (blocked — see ../toolchain-status.md)
1. **Not compiled, not run, no APK.** Android SDK IS installed now
   (platform-tools 37.0.3, platforms;android-34 rev 3, build-tools
   34.0.0), Gradle 8.7 is present, and the Gradle proxy config is
   written (`~/.gradle/gradle.properties`). The user changed
   `other_tcp` Deny -> Ask on 2026-11-21 17:07 EET, and the daemon
   NOW STARTS (pid 45318, listening on 127.0.0.1:41757) — but the
   build attempt on 2026-11-21 ~17:24 EET still failed: the Gradle
   client could not connect to its own daemon
   ("A new daemon was started but could not be connected to"), and
   the loopback connection's user approval timed out unanswered
   after ~10 minutes. No `gradlew` script exists yet (only
   `gradle-wrapper.properties`), no APK exists, no build log exists
   beyond the daemon log (`/root/.gradle/daemon/8.7/daemon-45318.out.log`).
   DO NOT retry the build until the user approves the daemon's
   loopback connection when it is asked, or switches `other_tcp` to
   Allow, or the build moves to another machine with Android Studio /
   Android SDK. No phone is connected here.
2. **SDK key: PRESENT (Development key in local.properties).**
   The Development apps key is in `local.properties` as
   `QURANLAB_KEY=qlk_v1_...` (git-ignored; never in source, never in
   chat) and is injected into `BuildConfig` (see `app/build.gradle.kts`).
   The real gateway reads it ONLY from BuildConfig and fails loudly
   if it is absent. The key binds to `org.quranapp.poc.harness` on the
   app's first launch with it (the portal asks for no applicationId).
   Signing certificates (SHA-256) are NOT yet added on the key's page
   at sdk.quranlab.ai — installs run on the CPU without hardware
   verification until they are. Also unverified: the Huawei Nova 5T
   bootloader status (an unlocked bootloader cannot open the model).
   Setup-day checklist (the build cannot settle these):
   - (a) On the key's page at sdk.quranlab.ai (signed in as
     @aiwithdranas) -> Keys -> "Quran Tasmee Test" (key ID
     kvgkzcyvkviv) -> "Signing certificates (SHA-256)" -> add the
     SHA-256 of the debug signing certificate the APK was built
     with (`keytool -list -v -keystore ~/.android/debug.keystore
     -alias androiddebugkey -storepass android`, or the release
     keystore's certificate if a release build is used instead).
     Until this is added, the model runs on the CPU without hardware
     verification.
   - (b) On the Huawei Nova 5T: check the bootloader is LOCKED
     (Developer options -> OEM unlocking must be OFF, and the
     bootloader state must read "locked"). An unlocked bootloader
     cannot open the model at all. Check BEFORE the first live
     session; if it is unlocked, the PoC cannot run on that phone.
   - (c) Record from the phone's Settings on setup day: Android
     version, RAM, and free storage.
3. **SDK version: 4.0.1 — CONFIRMED AND INSERTED.** The official SDK
   4.0 Android quick start (sdk.quranlab.ai/docs/4/quickstart/android)
   gives `implementation("ai.quranlab:tajweedformer:4.0.1")`, and the
   4.0.1 AAR is verified (8,742,966 bytes). SDK 3.x is NOT used.
4. **`assets/reference_67_32.json` EXISTS — VERIFIED 60/60 on
   2026-11-21 (build 14).** Generated from the
   quran-tajweed-phonetics dataset (60 ayahs, 4,146 units, SHA-256 in
   `assets/reference_67_32.meta.json`). The first version had 4,578
   units: it emitted a short vowel AND its madd letter as two units,
   but the model's tj3 merges them into ONE long vowel (fatha+alef ->
   aa, kasra+yeh -> ii, damma+waw -> uu — quran-g2p
   tokenlayer_tj3.py `_emit`). The launch check in MainActivity
   caught the mismatch on all 60 ayahs (build 13); the generator was
   fixed (432 madd merges, 0 quality mismatches) and the check now
   reports 60/60 against the loaded model's `model.reference`.
   Scoring is disabled on any mismatch. See `../step0-findings.md`.

## Wiring — DONE 2026-11-21 (build step)

MainActivity now constructs the real gateway when the app was built
with a Quran Lab key, the stub otherwise — the README's one-line
swap, with one necessary change: it is wrapped in `by lazy`,
because `applicationContext` is only available after the system
attaches the Activity, and a plain field initializer would crash
at Activity instantiation. The swap expression itself is unchanged:

```kotlin
private val gateway: TajweedSdkGateway by lazy {
    if (BuildConfig.HAS_QURANLAB_KEY) RealTajweedGateway(applicationContext)
    else StubTajweedGateway()
}
```

Also done in the same step: the LLR alignment in Alignment.kt
(fused log_probs + inter1_log_probs, weights/thresholds read from
the loaded model, balanced preset, R/U abstention, four-condition
slip guard, PoC metric definitions with the worked example as a
self-check); the calibration/heads adapter in RealTajweedGateway.kt
(`calibrationSourceOrNull()`, `headsDataOrNull()`,
`alignmentEvidenceOrNull()`); Stop cancels the session collection
job; `TajweedError` is caught around collection and shown as
code + message + fix. The session-end scoring call site
(MainActivity -> Bridge + calibration + align on the last finished
take's heads) is DONE (build 13, 2026-11-21): at take end the
harness scores the take against the expected ayah (67:1 by
default) with PROVISIONAL gray-zone margins (accept +2, reject -2,
slip +3 nats — to be frozen from the tuning set), logs per-word
verdicts + accuracy/recall, and saves the event log to the app's
private file `tasmee-event-log.txt` (the in-memory log alone is
lost on relaunch). The reference verification (item 3 below) is
DONE (build 14, 60/60).

First scored take (build 14, 2026-11-21 20:33 EET): user recited
67:1 correctly — score C=8 S=0 D=0 H=0 U=1 I=0 R=0, accuracy
100.0%, recall 100.0%, align_ms=319. Word 0 (تَبَارَكَ) abstained
U (llr -0.49, inside the provisional gray zone) — honest
abstention, not a false error; words 1-8 all C.

Ayah selection is DONE (build 16, 2026-11-21 20:40 EET):
surah buttons (الملك 67 / السجدة 32, jump to ayah 1) plus
prev/next steppers (wrap 1-30), in two rows — build 15 put all
five controls on one row and the next button ran off-screen.
Selection is locked while a take runs and until the reference is
verified; switching updates the expected text and logs
`expected_set S:A`. Verified live: tap on السجدة 32 showed
32:1 (الٓمٓ) with `expected_set 32:1` in the log. The score line
now uses the selected ayah (`expected.surah:expected.ayah`),
not the 67:1 constants.

Slip wiring is DONE in source (2026-11-22, not yet built — next
build): `Bridge.kt` gains `SlipCandidatesLoader` (parses the bundled
`assets/slip_candidates.json`, 18 ayahs, into "surah:ayah" ->
words' unit lists); MainActivity loads it at launch and verifies it
against the loaded model's `model.reference` exactly like the main
reference, logging `slip_check OK 18/18` on success, and passes it
to `Alignment.align` at take end. On any load/verify failure slip
detection stays off and ordinary word scoring is unaffected. The
file was checked 2026-11-22 against the approved planting tables:
all 14 slip targets are present (4 tuning + 10 test, including the
three replacements approved 2026-11-21: 16:78, 67:19, 32:28). The
phone-side check (`slip_check` in the event log) still has to be
seen live on the next build — until then slip sessions must not run.

Build-step checklist before the first live session:
1. Unblock the build: the user changed `other_tcp` Deny -> Ask on
   2026-11-21 17:07 EET and the daemon now starts, BUT the
   client->daemon loopback connection still needs a user approval
   that timed out on the 2026-11-21 ~17:24 EET attempt (see item 1
   above). DO NOT retry until the user approves that connection
   when asked, switches `other_tcp` to Allow, or the build moves to
   another machine with Android Studio / Android SDK.
2. `./gradlew :app:assembleDebug` (no `gradlew` script in this tree
   yet — generate it with `gradle wrapper` once the daemon can run).
3. Verify `reference_67_32.json` unit-for-unit against the loaded
   model's `model.reference` via `RealTajweedGateway.referenceUnits()`
   and re-generate on any mismatch; only then freeze it. — DONE
   (build 14, 2026-11-21: 60/60; the first version mismatched all
   60 on the madd merge, see item 4 above).
4. Read the `balanced` thresholds from the loaded model's
   `model.calibration` (never copy them into code); complete the LLR
   alignment in Alignment.kt against `lastFinalHeads`. — DONE (see
   above); thresholds are read at scoring time, not build time.
5. Wire the gateway (snippet above); make Stop end the take
   (`session.finish()` is already called by the gateway when its flow
   ends — MainActivity must cancel its collection on Stop) and catch
   `TajweedError` around collection, showing code + fix. — DONE.
6. First launch on Wi-Fi (model download ~55.5 MB, once), grant
   microphone permission, `adb install -r` on the Huawei Nova 5T
   (check its bootloader is locked first — see setup-day checklist
   in item 2 above), then tuning set -> freeze thresholds -> test set
   once, blind. See `../planting-tables.md`
   and the PoC document for the protocol. No live session is valid
   through the STUB gateway.

## Build (once unblocked)
```
cd harness
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```
Then: first launch on Wi-Fi (model download ~55.5 MB), grant microphone
permission, run the tuning set first, freeze thresholds, then run the
test set once, blind. See `../planting-tables.md` and the PoC document
for the protocol. No live session is valid through the STUB gateway.
