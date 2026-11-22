package org.quranapp.poc.harness

/**
 * PoC Harness — Stage 1 ONLY. Temporary single-screen test app.
 *
 * Screen: expected text (one ayah at a time), word highlighting during
 * tasmee', Start / Stop buttons, event log. No Mushaf display, no cards,
 * no full-app database, no Play release — none of that is built here.
 *
 * Flow: on launch the bundled reference_67_32.json is compared
 * unit-for-unit against the loaded model's own `model.reference`
 * (see Bridge.kt); scoring is disabled on any mismatch. The user taps
 * Start, recites the shown ayah (67:1 by default); the gateway streams
 * phonemes; at take end [Alignment] classifies each expected word as
 * C / S / D / H / U / I / R (see Alignment.kt) and the verdicts plus
 * the accuracy/recall summary are appended to the event log. The log
 * is also saved to the app's private file `tasmee-event-log.txt`
 * (the in-memory log alone is lost on relaunch). All scoring is
 * recomputed later from the raw event log — the UI never invents a
 * score.
 */
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import ai.quranlab.tajweedformer.TajweedError
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    // Wiring (README "Wiring"): the real gateway when the app was built
    // with a Quran Lab key, the stub otherwise (no recognition).
    // `by lazy` because the README's one-line swap passes
    // `applicationContext`, which is only available after the system
    // attaches the Activity — a plain field initializer would crash.
    // The swap itself is exactly the README's:
    //   if (BuildConfig.HAS_QURANLAB_KEY) RealTajweedGateway(applicationContext)
    //   else StubTajweedGateway()
    private val gateway: TajweedSdkGateway by lazy {
        if (BuildConfig.HAS_QURANLAB_KEY) RealTajweedGateway(applicationContext)
        else StubTajweedGateway()
    }
    private var sessionJob: Job? = null
    private val eventLog = EventLog()
    private val alignment = Alignment()

    /** Bundled reference (loaded + verified at launch); null until then. */
    private var bridge: ReferenceBridge? = null

    /** True only when the bundled JSON matched model.reference 60/60. */
    private var referenceVerified = false

    /** Slip candidates ("surah:ayah" -> that ayah's words' unit lists),
     *  loaded and verified against model.reference at launch; empty
     *  until then, so slip detection is off rather than wrong. */
    private var slipCandidates: Map<String, List<List<Int>>> = emptyMap()

    /** The ayah the current take is scored against (67:1 by default). */
    private var expectedAyah: ExpectedAyah? = null

    private val requestAudio =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { /* result handled on Start */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { HarnessScreen() }
    }

    /**
     * Saves the whole event log to the app's private file
     * `tasmee-event-log.txt` (best-effort: a failure here never breaks
     * a session; the on-screen log stays the live record).
     */
    private fun persistLog() {
        val snapshot = eventLog.lines()
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                applicationContext.openFileOutput(
                    "tasmee-event-log.txt", Context.MODE_PRIVATE,
                ).use { out ->
                    out.write(snapshot.joinToString("\n").toByteArray(Charsets.UTF_8))
                }
            } catch (_: Exception) {
                // Best-effort only; the screen log is unaffected.
            }
        }
    }

    @Composable
    private fun HarnessScreen() {
        var running by remember { mutableStateOf(false) }
        var status by remember {
            mutableStateOf(
                if (gateway.isStub) "STUB MODE — no SDK, no recognition. See README.md."
                else "Ready — SDK 4.0.1, Development key present. First launch downloads the model (~55.5 MB) on Wi-Fi.",
            )
        }
        var expectedText by remember { mutableStateOf("— loading reference… —") }
        var scoreText by remember { mutableStateOf("") }
        var logLines by remember { mutableStateOf(listOf<String>()) }
        var selectedSurah by remember { mutableStateOf(EXPECTED_SURAH) }
        var selectedAyah by remember { mutableStateOf(EXPECTED_AYAH) }

        val real = gateway as? RealTajweedGateway

        /** Switch the ayah the next take is scored against (tuning/test sets). */
        fun selectAyah(surah: Int, ayah: Int) {
            val b = bridge ?: return
            if (running) return
            val next = try {
                b.expectedAyah(surah, ayah)
            } catch (_: Exception) {
                return
            }
            selectedSurah = surah
            selectedAyah = ayah
            expectedAyah = next
            expectedText = next.text
            scoreText = ""
            status = "Expected: $surah:$ayah."
            eventLog.append("expected_set $surah:$ayah")
            logLines = eventLog.lines()
            persistLog()
        }

        // Launch check: load the bundled reference and verify it
        // against the loaded model's own model.reference. Scoring stays
        // disabled unless the check passes 60/60.
        LaunchedEffect(Unit) {
            if (real == null) return@LaunchedEffect
            try {
                val loaded = ReferenceLoader.loadFromAssets(applicationContext)
                bridge = loaded
                val model = real.prepare()
                val check = verifyReferenceAgainstModel(
                    loaded,
                    { s, a -> real.referenceUnits(s, a) },
                    model.tokens.wordBar,
                )
                if (check.mismatches.isEmpty()) {
                    referenceVerified = true
                    expectedAyah = loaded.expectedAyah(EXPECTED_SURAH, EXPECTED_AYAH)
                    expectedText = expectedAyah?.text ?: "—"
                    status = "Reference verified ${check.matched}/${check.total} — ready. " +
                        "Expected: $EXPECTED_SURAH:$EXPECTED_AYAH."
                    eventLog.append("reference_check OK ${check.matched}/${check.total}")
                    // Slip candidates: load the bundled file and verify
                    // it against the model's own model.reference exactly
                    // like the main reference. On any failure slip
                    // detection stays off (empty map); ordinary word
                    // scoring is unaffected.
                    try {
                        val loadedSlip = SlipCandidatesLoader.loadFromAssets(applicationContext)
                        val slipBridge = ReferenceBridge(
                            loadedSlip.values.associate {
                                Pair(it.ayah.surah, it.ayah.ayah) to it.ayah
                            },
                        )
                        val slipCheck = verifyReferenceAgainstModel(
                            slipBridge,
                            { s, a -> real.referenceUnits(s, a) },
                            model.tokens.wordBar,
                        )
                        if (slipCheck.mismatches.isEmpty()) {
                            slipCandidates = loadedSlip.mapValues { it.value.words }
                            eventLog.append("slip_check OK ${slipCheck.matched}/${slipCheck.total}")
                        } else {
                            eventLog.append(
                                "slip_check MISMATCH (${slipCheck.mismatches.size}) — slip detection disabled. See log.",
                            )
                        }
                    } catch (e: Exception) {
                        eventLog.append("slip_check error: ${e.message} — slip detection disabled.")
                    }
                } else {
                    status = "Reference MISMATCH (${check.mismatches.size}) — scoring disabled. See log."
                    eventLog.append(
                        "reference_check MISMATCH: ${check.mismatches.joinToString("; ")}",
                    )
                }
            } catch (e: Exception) {
                status = "Reference check failed: ${e.message}"
                eventLog.append("reference_check error: ${e.message}")
            }
            logLines = eventLog.lines()
            persistLog()
        }

        Column(Modifier.fillMaxSize().padding(16.dp)) {
            Text("Tasmee PoC Harness — STAGE 1 ONLY", fontSize = 18.sp, fontWeight = FontWeight.Bold)
            Text(status, fontSize = 14.sp)
            Spacer(Modifier.height(12.dp))
            Text(buildAnnotatedString {
                // Word highlighting: the current word is rendered bold.
                // In stub mode there is no live word; this is the UI hook.
                withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(expectedText) }
            }, fontSize = 20.sp)
            if (scoreText.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                Text(scoreText, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.height(12.dp))
            // Ayah selection (tuning set 67:1-12 + 32:1-8, test set the
            // rest). Locked while a take runs and until the reference
            // is verified — scoring needs the verified reference.
            // Two rows: the five controls do not fit on one phone row.
            Row {
                Button(
                    onClick = { selectAyah(67, 1) },
                    enabled = !running && referenceVerified,
                ) { Text("الملك 67") }
                Spacer(Modifier.width(8.dp))
                Button(
                    onClick = { selectAyah(32, 1) },
                    enabled = !running && referenceVerified,
                ) { Text("السجدة 32") }
            }
            Spacer(Modifier.height(8.dp))
            Row {
                Button(
                    onClick = {
                        selectAyah(selectedSurah, (selectedAyah - 2 + 30) % 30 + 1)
                    },
                    enabled = !running && referenceVerified,
                ) { Text("◀ السابقة") }
                Spacer(Modifier.width(8.dp))
                Text(
                    "$selectedSurah:$selectedAyah",
                    fontSize = 18.sp, fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(top = 8.dp),
                )
                Spacer(Modifier.width(8.dp))
                Button(
                    onClick = { selectAyah(selectedSurah, selectedAyah % 30 + 1) },
                    enabled = !running && referenceVerified,
                ) { Text("التالية ▶") }
            }
            Spacer(Modifier.height(12.dp))
            Row {
                Button(onClick = {
                    if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.RECORD_AUDIO)
                        != PackageManager.PERMISSION_GRANTED
                    ) {
                        requestAudio.launch(Manifest.permission.RECORD_AUDIO)
                        return@Button
                    }
                    running = true
                    status = if (gateway.isStub) "Listening (stub: no model)…" else "Listening…"
                    scoreText = ""
                    eventLog.append("session_start")
                    sessionJob = lifecycleScope.launch {
                        var lastChunk: HeardChunk? = null
                        var sessionError: TajweedError? = null
                        try {
                            gateway.startSession().collect { chunk ->
                                // Real gateway: chunk.decoded.tokens -> Alignment.
                                // Stub gateway: emits nothing; this block is the
                                // integration point, verified by unit tests with
                                // synthetic phoneme streams.
                                lastChunk = chunk
                                eventLog.append("chunk: ${chunk.summary}")
                                logLines = eventLog.lines()
                            }
                        } catch (e: TajweedError) {
                            // TajweedError carries code + fix: show both,
                            // never a bare stack trace.
                            sessionError = e
                        } catch (e: CancellationException) {
                            // Stop was pressed: the take was still finished
                            // by the gateway (see RealTajweedGateway), so
                            // scoring below still runs on its heads.
                            eventLog.append("session_cancelled")
                        }
                        // The take finished (or the session ended by itself):
                        // the gateway kept the whole-take heads for the
                        // LLR alignment (see Alignment.kt).
                        val heads = real?.lastFinalHeads
                        eventLog.append(
                            if (heads != null) {
                                val names = heads.names.joinToString(",")
                                val lp = runCatching { heads["log_probs"] }.getOrNull()
                                "take_finished heads=[$names] frames=${lp?.frames ?: "?"}"
                            } else {
                                "take_finished heads=none"
                            },
                        )
                        // Raw word-boundary evidence before any alignment decision.
                        // Written to tasmee-event-log.txt (not Android Logcat).
                        if (real != null) {
                            val boundaryDiagnostics = real.wordBoundaryDiagnosticsOrNull()
                            eventLog.append(
                                "word_boundary_raw ${boundaryDiagnostics ?: "unavailable"}",
                            )
                        }
                        // --- Session-end scoring (Stage 1 wiring) ---
                        // Runs only when the reference was verified; the
                        // margins are PROVISIONAL (see companion) until
                        // the tuning set freezes them.
                        val expected = expectedAyah
                        if (real != null && referenceVerified && expected != null) {
                            val evidence = real.alignmentEvidenceOrNull()
                            val finalDecoded = real.lastFinalDecoded
                            val heardIds = finalDecoded?.tokens?.map { it.id }
                                ?: lastChunk?.tokenIds
                            val sdkHeardWords = finalDecoded?.words?.map { word ->
                                word.tokens.map { token -> token.id }
                            }
                            if (finalDecoded != null) {
                                // Evidence can be unavailable even when decoding succeeded.
                                // Log the boundary count only when its ID is known.
                                val barCountText = evidence?.heads?.wordBarId?.let { barId ->
                                    finalDecoded.tokens.count { it.id == barId }.toString()
                                } ?: "unknown"
                                eventLog.append(
                                    "heard_segmentation tokens=${finalDecoded.tokens.size} " +
                                        "wordBars=$barCountText sdkWords=${finalDecoded.words.size} " +
                                        "expectedWords=${expected.wordIndex.distinct().size}",
                                )
                            }
                            if (evidence == null || heardIds == null) {
                                eventLog.append(
                                    "score_unavailable evidence=${evidence != null} " +
                                        "tokens=${heardIds != null}",
                                )
                            } else {
                                try {
                                    val t0 = System.currentTimeMillis()
                                    val judgements = alignment.align(
                                        expected, heardIds,
                                        acceptThreshold = ACCEPT_MARGIN,
                                        rejectThreshold = REJECT_MARGIN,
                                        slipMargin = SLIP_MARGIN,
                                        evidence = evidence,
                                        slipCandidates = slipCandidates,
                                        wordBarId = evidence.heads.wordBarId,
                                        heardWordTokenIds = sdkHeardWords,
                                    )
                                    val ms = System.currentTimeMillis() - t0
                                    val counts = Alignment.counts(judgements)
                                    val c = counts[WordVerdict.C] ?: 0
                                    val s = counts[WordVerdict.S] ?: 0
                                    val d = counts[WordVerdict.D] ?: 0
                                    val h = counts[WordVerdict.H] ?: 0
                                    val u = counts[WordVerdict.U] ?: 0
                                    val i = counts[WordVerdict.I] ?: 0
                                    val r = counts[WordVerdict.R] ?: 0
                                    val n = c + s + d + h + u
                                    val acc = Alignment.verbalAccuracy(c, s, d, i)
                                    val rec = Alignment.independentRecall(c, n, h, u)
                                    val accText = acc?.let { "%.1f".format(it * 100) } ?: "?"
                                    val recText = rec?.let { "%.1f".format(it * 100) } ?: "?"
                                    eventLog.append(
                                        "score ${expected.surah}:${expected.ayah} " +
                                            "C=$c S=$s D=$d H=$h U=$u I=$i R=$r " +
                                            "accuracy=$accText% recall=$recText% " +
                                            "align_ms=$ms (PROVISIONAL margins " +
                                            "accept=+$ACCEPT_MARGIN reject=$REJECT_MARGIN " +
                                            "slip=+$SLIP_MARGIN — freeze from tuning set)",
                                    )
                                    for (j in judgements) {
                                        val llr = j.fusedLlr?.let { "%.2f".format(it) } ?: "-"
                                        val alt = j.alternativeAyah?.let { " alt=$it" } ?: ""
                                        eventLog.append("word ${j.wordIndex} ${j.verdict} llr=$llr$alt")
                                    }
                                    scoreText = "Score: C=$c S=$s D=$d U=$u I=$i R=$r — " +
                                        "accuracy $accText% — recall $recText%"
                                } catch (e: Exception) {
                                    eventLog.append("score_error: ${e.message}")
                                }
                            }
                        } else {
                            eventLog.append(
                                "score_skipped verified=$referenceVerified " +
                                    "expected=${expected != null}",
                            )
                        }
                        logLines = eventLog.lines()
                        persistLog()
                        status = if (sessionError != null) {
                            "Error ${sessionError.code}: ${sessionError.message} — Fix: ${sessionError.fix}"
                        } else if (heads != null) {
                            "Take finished — scored. See log."
                        } else {
                            "Take finished — heads not available."
                        }
                        if (sessionError != null) {
                            eventLog.append(
                                "error ${sessionError.code}: ${sessionError.message} " +
                                    "(fix: ${sessionError.fix})",
                            )
                            logLines = eventLog.lines()
                            persistLog()
                        }
                        running = false
                    }
                }, enabled = !running) { Text("Start") }
                Spacer(Modifier.width(8.dp))
                Button(onClick = {
                    sessionJob?.cancel()
                    sessionJob = null
                    running = false
                    status = "Stopped."
                    eventLog.append("session_stop")
                    logLines = eventLog.lines()
                    persistLog()
                }, enabled = running) { Text("Stop") }
            }
            Spacer(Modifier.height(12.dp))
            Text("Event log (raw — scores are recomputed from this log):", fontWeight = FontWeight.Bold)
            LazyColumn(Modifier.weight(1f)) {
                items(logLines) { line -> Text(line, fontSize = 13.sp) }
            }
            Text("SDK key present: ${BuildConfig.HAS_QURANLAB_KEY} (key is never shown)", fontSize = 12.sp)
        }
    }

    companion object {
        /** The ayah the harness scores takes against (first scored ayah). */
        const val EXPECTED_SURAH = 67
        const val EXPECTED_AYAH = 1

        /**
         * PROVISIONAL gray-zone margins (nats, relative to the
         * calibration threshold): margin >= +2 confirms the alternative
         * (S/D/I), margin <= -2 confirms the expected word (C), between
         * = U (abstain). Slip margin: the alternative reading must beat
         * the expected reading by +3 nats over the window. These are NOT
         * frozen: the tuning set chooses them; the test set runs them
         * frozen. Never present a score from these margins as final.
         */
        const val ACCEPT_MARGIN = 2.0
        const val REJECT_MARGIN = -2.0
        const val SLIP_MARGIN = 3.0
    }
}
