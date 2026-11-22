package org.quranapp.poc.harness

import ai.quranlab.tajweedformer.TajweedCalibration
import ai.quranlab.tajweedformer.TajweedClient
import ai.quranlab.tajweedformer.TajweedClientOptions
import ai.quranlab.tajweedformer.TajweedDecoded
import ai.quranlab.tajweedformer.TajweedError
import ai.quranlab.tajweedformer.TajweedEvent
import ai.quranlab.tajweedformer.TajweedHead
import ai.quranlab.tajweedformer.TajweedHeads
import ai.quranlab.tajweedformer.TajweedModel
import ai.quranlab.tajweedformer.TajweedSessionOptions
import ai.quranlab.tajweedformer.TajweedTokens
import android.content.Context
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Real gateway to the Quran Lab SDK 4.0 — Stage 1 ONLY.
 *
 * API surface (verified 2026-11-21 against the installed 4.0.1 AAR and
 * Quran Lab's own docs; do not guess other names):
 * - SDK 4.0 Android quick start: https://sdk.quranlab.ai/docs/4/quickstart/android
 *   `implementation("ai.quranlab:tajweedformer:4.0.1")`
 * - Full API reference for coding assistants:
 *   https://github.com/Quran-Lab/tajweedformer-android/blob/main/AGENTS.md
 * - Feedback / heads / calibration / reference method:
 *   https://sdk.quranlab.ai/docs/feedback
 * - Example app (the four calls, error handling, sample clips):
 *   https://github.com/Quran-Lab/tajweedformer-android/blob/main/examples/quickstart/src/main/kotlin/ai/quranlab/tajweedformer/example/ExampleActivity.kt
 *
 * The four calls, exactly as documented:
 *   1. `TajweedClient(context, key)` — top-level function in
 *      `ai.quranlab.tajweedformer` (TajweedClientAndroid.kt), one Client
 *      per process; a second client while the first is not closed is the
 *      error `client_exists`. Key comes ONLY from BuildConfig (see below).
 *   2. `client.model()` — first call downloads and verifies the model
 *      (55.5 MB, once, 68.4 MB on disk); `client.prefetch()` warms it.
 *   3. `model.listen(TajweedSessionOptions(...))` — microphone take;
 *      `model.session()` + `session.push(samples)` replays own audio.
 *   4. `session.chunks.collect { chunk -> ... }` — every 320 ms a chunk
 *      with `chunk.decoded` (text, tokens, words, settled, isFinal) and
 *      `chunk.heads` (the raw network outputs selected for the take);
 *      `session.finish()` returns the take's whole `TajweedHeads`.
 *
 * KEY FLOW (the only copy of the key in the app):
 *   local.properties (git-ignored, `QURANLAB_KEY=qlk_v1_...`)
 *     -> app/build.gradle.kts reads it into BuildConfig.QURANLAB_KEY /
 *        BuildConfig.HAS_QURANLAB_KEY
 *     -> this file reads BuildConfig ONLY, at client creation.
 * The key is never hardcoded, never logged, never shown, never copied
 * anywhere else. If it is absent/blank, client creation throws
 * IllegalStateException with a clear message instead of proceeding.
 *
 * WHAT THIS FILE DOES NOT DO (build-step work, not skipped — deferred):
 * - It does NOT wire itself into MainActivity. MainActivity still
 *   constructs `StubTajweedGateway()`; the build step swaps that one
 *   line for `RealTajweedGateway(applicationContext)` once the
 *   reference file is verified and the session lifecycle (Stop ->
 *   finish, thresholds, reference gate) is in place. See README.md.
 * - It does NOT implement the LLR alignment (Alignment.kt is still the
 *   stub that returns U for every word). It exposes the raw material
 *   the alignment needs: per-chunk token ids + confidences in
 *   [HeardChunk], the whole-take heads in [lastFinalHeads] after the
 *   flow completes, and the loaded model's [calibration] and
 *   [referenceUnits] (read from the loaded model, never copied).
 * - It does NOT judge a recitation. The SDK is inference-only; Quran
 *   Lab ships no scorer. Judgement is the Harness's Alignment work.
 *
 * HEADS SELECTED for every take: the documented default computed set —
 * `log_probs`, `inter1_log_probs`, `timing_log_probs`, `gem`,
 * `attr_qalqalah`, `conf`. The LLR method (docs/feedback) fuses
 * `log_probs` + `inter1_log_probs`; the other heads are recorded raw
 * for the build step. `attr_ghunna` / `attr_tafkheem` have no validated
 * thresholds in this release and are NOT selected (raw signals only).
 */
class RealTajweedGateway(
    private val context: Context,
) : TajweedSdkGateway {

    override val isStub: Boolean = false

    private val modelMutex = Mutex()

    @Volatile
    private var cachedClient: TajweedClient? = null

    @Volatile
    private var cachedModel: TajweedModel? = null

    /**
     * Whole-take heads of the most recent finished take, or null if no
     * take has finished yet. Set by [startSession] when the take ends
     * (via `session.finish()`). This is the input the LLR alignment
     * reads: `log_probs` and `inter1_log_probs` as float32
     * [frames, classes], row-major — `heads["log_probs"].values`,
     * `heads["log_probs"].frames`, `.classes`, `.frameMs`.
     * Volatile: written on the session coroutine, read by the build
     * step's scoring code after the flow completes.
     */
    @Volatile
    var lastFinalHeads: TajweedHeads? = null
        private set

    /** Final decoded layer of the most recent finished take, if any. */
    @Volatile
    var lastFinalDecoded: TajweedDecoded? = null
        private set

    /** Last session event seen (SpeechStarted / SpeechEnded / Ended …). */
    @Volatile
    var lastEvent: TajweedEvent? = null
        private set

    /** Model version of the loaded model, e.g. "4.0.1"-series build. */
    @Volatile
    var lastModelVersion: String? = null
        private set

    /**
     * The one Client of the process, created lazily from the BuildConfig
     * key. Fails loudly (IllegalStateException) when the key is absent.
     * If a Client already exists in this process (a second gateway
     * instance), reuses `TajweedClient.current` instead of throwing
     * `client_exists`.
     */
    private fun client(): TajweedClient {
        cachedClient?.let { return it }
        synchronized(this) {
            cachedClient?.let { return it }
            if (!BuildConfig.HAS_QURANLAB_KEY || BuildConfig.QURANLAB_KEY.isBlank()) {
                throw IllegalStateException(
                    "REAL GATEWAY: no Quran Lab SDK key. " +
                        "Put the Development apps key in harness/local.properties " +
                        "as QURANLAB_KEY=qlk_v1_... (git-ignored) and rebuild. " +
                        "The key is never shown here and never belongs in source."
                )
            }
            val created = try {
                TajweedClient(
                    context.applicationContext,
                    BuildConfig.QURANLAB_KEY,
                    TajweedClientOptions(unmeteredOnly = true),
                )
            } catch (e: TajweedError) {
                if (e.code == "client_exists") {
                    TajweedClient.current
                        ?: throw IllegalStateException(
                            "REAL GATEWAY: a Client already exists but " +
                                "TajweedClient.current is null (code client_exists).",
                            e,
                        )
                } else {
                    throw e
                }
            }
            cachedClient = created
            return created
        }
    }

    /**
     * Downloads (first run only, ~55.5 MB, on Wi-Fi because
     * `unmeteredOnly = true`) and opens the model, then caches it.
     * [onDownloadProgress] receives (doneBytes, totalBytes) while the
     * first download runs; it is not called again once the model is on
     * the device. Throws TajweedError (code, message, fix, docs) on
     * failure — e.g. `invalid_key`, `needs_connection`,
     * `download_failed`. The caller (build step) shows code + fix.
     */
    suspend fun prepare(
        onDownloadProgress: (doneBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): TajweedModel {
        cachedModel?.let { return it }
        return modelMutex.withLock {
            cachedModel ?: run {
                val c = client()
                c.prefetch(unmeteredOnly = true) { done, total ->
                    onDownloadProgress(done, total)
                }
                c.model().also {
                    cachedModel = it
                    lastModelVersion = it.version
                }
            }
        }
    }

    /** The loaded model, or null if [prepare] has not run yet. */
    fun modelOrNull(): TajweedModel? = cachedModel

    /**
     * The loaded model's vocabulary: 38 tokens. `tokens.wordBar` is the
     * word-boundary token id; `tokens.name(id)` names a token id.
     * Null until [prepare] has run.
     */
    fun tokensOrNull(): TajweedTokens? = cachedModel?.tokens

    /**
     * The loaded model's calibration (`calibration.json` exactly as
     * shipped: `calibration.json`, `sha256`, format
     * `quranlab.tajweed-calibration/1`). Null when the build has none.
     * Read thresholds from it — never copy them into code:
     * `calibration.threshold(ctx, class, preset)` (nats on the fused
     * LLR), `calibration.fusionWeight(ctx, class)` (the weight `a` in
     * `a * llr(log_probs) + (1 - a) * llr(inter1_log_probs)`),
     * `calibration.status(ctx, class)` (`validated` / `experimental` /
     * `off`). PoC uses the `balanced` preset unless the tuning set
     * proves otherwise.
     */
    fun calibrationOrNull(): TajweedCalibration? = cachedModel?.calibration

    /**
     * Expected unit ids of one ayah from the loaded model's own
     * reference data (`model.reference`), with the word bar
     * (`tokens.wordBar`) between words. This is the authoritative
     * source the bundled `reference_67_32.json` (PROVISIONAL) must be
     * compared against unit-for-unit before any session is frozen.
     * In SDK 4.0 the reference ships with the model; `fetch()` is a
     * no-op once downloaded (`isDownloaded` true). Throws
     * IllegalStateException when the build carries no reference data,
     * and TajweedError (`reference_missing`, `invalid_app_option`)
     * from the SDK for a missing lookup.
     */
    suspend fun referenceUnits(surah: Int, ayah: Int): IntArray {
        val model = prepare()
        val reference = model.reference
            ?: throw IllegalStateException(
                "REAL GATEWAY: this model build carries no reference " +
                    "data (model.reference is null); cannot verify the " +
                    "expected units of $surah:$ayah.",
            )
        reference.fetch()
        return reference.units(surah, ayah)
    }

    /**
     * SDK self-diagnosis as JSON (safe to paste into a bug report):
     * SDK version, whether the key was accepted and why not, model on
     * device and whether it verified, backend, microphone permission,
     * storage, last error, connection. Never throws; works offline.
     */
    suspend fun diagnoseJson(): String = client().diagnose().toJson()

    /**
     * One live microphone take. The SDK asks for RECORD_AUDIO itself
     * (a refusal is TajweedError `microphone_denied`); the take ends
     * by itself 1.5 s after the reciter stops (`endOnSilenceMs = 1500`,
     * the quick-start value), or at `session.finish()` / the 20-minute
     * limit. Each emitted [HeardChunk] carries the decoded tokens so
     * far (`tokenIds` in `model.tokens`, parallel `confidences`) plus
     * `summary` (decoded text) and `isFinal` on the last chunk; the
     * first `settled` tokens will not change any more.
     *
     * When the flow completes, [lastFinalHeads] holds the whole-take
     * heads (via `session.finish()`) for the build step's LLR
     * alignment. If the collector is cancelled, the take is finished
     * the same way when possible, else cancelled. TajweedError with
     * code `cancelled` is benign and swallowed; any other TajweedError
     * propagates to the collector (the build step's MainActivity must
     * catch it and show code + fix — it carries both).
     *
     * NOTE: this flow does NOT judge anything. Verdicts come from
     * Alignment.kt once it is completed against [lastFinalHeads].
     */
    override fun startSession(): Flow<HeardChunk> = flow {
        val model = prepare()
        val session = model.listen(
            TajweedSessionOptions(
                endOnSilenceMs = END_ON_SILENCE_MS,
                heads = SELECTED_HEADS,
            ),
        )
        var finished = false
        try {
            // Events run beside the chunks in a child coroutine (needed
            // for the interruption tests: INTERRUPTION / ROUTE_CHANGE /
            // Ended). The child is cancelled as soon as the chunks end;
            // `flow {}` itself is not a CoroutineScope, hence the
            // explicit coroutineScope.
            coroutineScope {
                val eventsJob = launch {
                    session.events.collect { event -> lastEvent = event }
                }
                try {
                    session.chunks.collect { chunk ->
                        val decoded = chunk.decoded
                        emit(
                            HeardChunk(
                                summary = decoded.text,
                                tokenIds = decoded.tokens.map { it.id },
                                confidences = decoded.tokens.map { it.confidence.toDouble() },
                                isFinal = decoded.isFinal,
                            ),
                        )
                    }
                } finally {
                    eventsJob.cancel()
                }
            }
        } catch (e: TajweedError) {
            if (e.code != "cancelled") throw e
        } finally {
            // The take must be finished (or cancelled) even when the
            // collector goes away: finish() is a suspend call, so it
            // runs under NonCancellable or cancellation would abort it.
            withContext(NonCancellable) {
                if (!finished) {
                    try {
                        val heads = session.finish()
                        lastFinalHeads = heads
                        lastFinalDecoded = heads.decoded()
                        finished = true
                    } catch (e: TajweedError) {
                        if (e.code != "cancelled") {
                            try {
                                session.cancel()
                            } catch (_: Exception) {
                                // Take is already over; nothing to cancel.
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Closes the model and the client. Call when the harness is done
     * (end of the app's life), never mid-session.
     */
    fun close() {
        try {
            cachedModel?.close()
        } catch (_: Exception) {
        }
        cachedModel = null
        try {
            cachedClient?.close()
        } catch (_: Exception) {
        }
        cachedClient = null
    }

    // ------------------------------------------------------------------
    // Calibration + heads access for the LLR alignment (Alignment.kt).
    // These read the loaded model only — no threshold, weight or
    // token value is copied into code.
    // ------------------------------------------------------------------

    /**
     * The loaded model's `calibration.json` as a [CalibrationSource],
     * or null when the model is not loaded. Nulls from the SDK are
     * passed through as "unknown", never invented.
     */
    fun calibrationSourceOrNull(): CalibrationSource? {
        val cal = cachedModel?.calibration ?: return null
        return object : CalibrationSource {
            override val contexts: List<String> = cal.contexts
            override val classes: List<String> = cal.classes
            override val presets: List<String> = cal.presets
            override val defaultContext: String = cal.defaultContext ?: ""
            override val defaultPreset: String = cal.defaultPreset ?: ""
            // Note: the SDK types these two as nullable, but the shipped
            // calibration always carries them. "" is a sentinel that
            // matches no real context/preset name, so any lookup with it
            // returns null (unknown) — never an invented judgment.
            override fun fusionWeight(context: String, klass: String): Double? =
                cal.fusionWeight(context, klass)

            override fun threshold(context: String, klass: String, preset: String): Double? =
                cal.threshold(context, klass, preset)

            override fun status(context: String, klass: String): String? =
                cal.status(context, klass)
        }
    }

    /**
     * Compact, read-only diagnostics for the word-boundary class in the
     * raw CTC heads. This does not change decoding, thresholds, or scoring.
     * Returns null if the SDK did not provide the required heads/classes.
     */
    fun wordBoundaryDiagnosticsOrNull(): String? {
        val allHeads = lastFinalHeads ?: return null
        val tokens = tokensOrNull() ?: return null
        val barId = tokens.wordBar
        val blankId = tokens.blank

        fun summarize(headName: String): String {
            val head = runCatching { allHeads[headName] }.getOrNull()
                ?: return "$headName=unavailable"
            if (head.frames <= 0 || barId !in 0 until head.classes || blankId !in 0 until head.classes) {
                return "$headName=unavailable(frames=${head.frames},classes=${head.classes},bar=$barId,blank=$blankId)"
            }
            var top1Bar = 0
            var top2Bar = 0
            var barAboveBlank = 0
            var maxBarMinusBlank = Float.NEGATIVE_INFINITY
            for (frame in 0 until head.frames) {
                val bar = head.get(frame, barId)
                val blank = head.get(frame, blankId)
                if (bar > blank) barAboveBlank++
                if (bar - blank > maxBarMinusBlank) maxBarMinusBlank = bar - blank

                var bestId = -1
                var best = Float.NEGATIVE_INFINITY
                var second = Float.NEGATIVE_INFINITY
                for (klass in 0 until head.classes) {
                    val value = head.get(frame, klass)
                    if (value > best) {
                        second = best
                        best = value
                        bestId = klass
                    } else if (value > second) {
                        second = value
                    }
                }
                if (bestId == barId) top1Bar++
                if (bestId == barId || (head.get(frame, barId) >= second)) top2Bar++
            }
            val greedyBars = runCatching { allHeads.greedy(headName).count { it.id == barId } }
                .getOrNull()?.toString() ?: "unavailable"
            val margin = if (maxBarMinusBlank.isFinite()) "%.4f".format(java.util.Locale.US, maxBarMinusBlank) else "unknown"
            return "$headName(frames=${head.frames},top1Bar=$top1Bar,top2Bar=$top2Bar,barGtBlank=$barAboveBlank,maxBarMinusBlank=$margin,greedyBars=$greedyBars)"
        }

        return "barId=$barId blankId=$blankId ${summarize("log_probs")} ${summarize("inter1_log_probs")}"
    }

    /**
     * Frame-level CTC log-probabilities of the last finished take,
     * or null when no take has finished. `inter1_log_probs` is
     * prepared here (word bar merged into blank) exactly as the
     * calibration's `inter1_prep` rule says; `log_probs` is used
     * exactly as the model emitted it.
     */
    fun headsDataOrNull(): CtcHeads? {
        val heads = lastFinalHeads ?: return null
        val tokens = tokensOrNull() ?: return null
        val lpHead = runCatching { heads["log_probs"] }.getOrNull() ?: return null
        val ilpHead = runCatching { heads["inter1_log_probs"] }.getOrNull() ?: return null
        val lp = headRows(lpHead) ?: return null
        val ilpRaw = headRows(ilpHead) ?: return null
        val blankId = tokens.blank
        val wordBarId = tokens.wordBar
        val ilp = prepareInter1LogProbs(ilpRaw, blankId, wordBarId)
        return CtcHeads(lp, ilp, lpHead.classes, blankId, wordBarId, tokens.names)
    }

    /**
     * Everything the LLR alignment needs from the last finished take,
     * or null when no take has finished or the model is not loaded.
     * Context is the model's default context; the preset is "balanced"
     * when the model offers it, otherwise the model's default preset
     * (the PoC uses the balanced preset).
     */
    fun alignmentEvidenceOrNull(): AlignmentEvidence? {
        val heads = headsDataOrNull() ?: return null
        val cal = calibrationSourceOrNull() ?: return null
        val preset = if ("balanced" in cal.presets) "balanced" else cal.defaultPreset
        return AlignmentEvidence(heads, cal, cal.defaultContext, preset)
    }

    /** One head's row-major [frames, classes] values as per-frame rows. */
    private fun headRows(head: TajweedHead): Array<FloatArray>? {
        if (head.frames <= 0 || head.classes <= 0) return null
        val values = head.values
        if (values.size != head.frames * head.classes) return null
        return Array(head.frames) { f ->
            values.copyOfRange(f * head.classes, (f + 1) * head.classes)
        }
    }

    companion object {
        /** Take ends by itself after this much silence (quick-start value). */
        const val END_ON_SILENCE_MS = 1500

        /**
         * Heads computed for every take: the documented default set —
         * the three CTC heads plus gemination, qalqalah and confidence.
         * `log_probs` + `inter1_log_probs` are the LLR fusion pair.
         */
        val SELECTED_HEADS: List<String> = listOf(
            "log_probs",
            "inter1_log_probs",
            "timing_log_probs",
            "gem",
            "attr_qalqalah",
            "conf",
        )
    }
}
