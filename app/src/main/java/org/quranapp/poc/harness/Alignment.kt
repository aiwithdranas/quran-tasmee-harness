package org.quranapp.poc.harness

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * Alignment + verdicts — implements the PoC's mathematical definitions
 * (quran-app-stage1-poc, "التعريف الرياضي لمقاييس التقييم") and the
 * LLR method documented by Quran Lab:
 * - docs/feedback ("Build your own feedback on the model's output"):
 *   "for each position of the expected sequence and each single-unit
 *   alternative of a class, compute the full-sequence CTC forward
 *   log-probability of the expected sequence and of the alternative,
 *   on `log_probs` and again on `inter1_log_probs` … fuse … clip …
 *   and compare it with the calibration's threshold for that context
 *   and class."
 * - tajweedformer-android AGENTS.md ("Calibration and reference data"):
 *   fusion `llr = a * llr(log_probs) + (1 - a) * llr(inter1_log_probs)`,
 *   `inter1_prep` (the word bar merged into blank), clip [-30, 30],
 *   classes `vowel_sub`, `vowel_del`, `vowel_ins`, `short_to_long`,
 *   `long_to_short`, `cons_sub`, `cons_del`, `insertion`, presets
 *   `strict`, `balanced`, `sensitive`, `very_sensitive`.
 * - tajweedformer-android docs/migrating-to-3.md §3: "Words. The
 *   largest clipped fused LLR over every single-unit edit inside the
 *   word, against the word level of `contexts.<ctx>.word`." A class
 *   whose `status` is `off` is not judged.
 *
 * One classification per expected word, exactly one of:
 *   C correct-confirmed, S substitution, D deletion, H hinted,
 *   U unclear (abstain), I insertion (extra heard word, no expected word),
 *   R repetition (repeat / restart / pause-then-resume — recorded, NEVER
 *     counted as an error and NEVER counted as an insertion).
 *
 * N = C + S + D + H + U   (prompt words shown in advance are excluded
 * from N entirely; R never enters any penalty denominator).
 * Verbal accuracy = C / (C + S + D + I)
 * Independent recall = C / (N - H - U)
 * If a denominator is 0, the result is "insufficient evidence", never 100%.
 *
 * Two evidence levels (same verdicts, different strength):
 * 1. HEADS path ([AlignmentEvidence] present): the full LLR method.
 *    Per word, the largest clipped fused LLR over every single-unit
 *    edit inside the word (migrating-to-3.md §3 item 4), each edit
 *    compared with its class threshold from the loaded model's
 *    calibration (`balanced` preset). The accept/reject pair is the
 *    frozen gray zone around that threshold, chosen on the TUNING set
 *    only: margin >= accept -> the alternative is confirmed (S/D/I);
 *    margin <= reject -> the expected word is confirmed (C);
 *    between -> U (abstain). Thresholds and fusion weights are NEVER
 *    copied into this file — they are read from the loaded model
 *    through [CalibrationSource].
 * 2. TOKEN-ONLY path (no evidence): the greedy decoded token ids
 *    alone. Exact matches -> C, missing words -> D, extra words -> I,
 *    repeats/restarts -> R. A word that differs from what was heard is
 *    U, never S: without LLR evidence this code never accuses a
 *    reciter of saying a different word. Live sessions MUST use the
 *    heads path; the token path exists for unit tests and synthetic
 *    streams.
 *
 * H (hinted) is never produced here: the harness has no hint
 * mechanism in a session. H counts are produced by the app layer
 * (a later stage) when a word is revealed as a hint.
 *
 * Abstention / mutashabihat guard (PoC "آلية الامتناع"):
 * a slip-to-a-similar-ayah verdict requires ALL FOUR conditions —
 * (1) every word in the window is above the accept threshold (no U
 *     word in the window), (2) the alternative reading holds for >= 3
 *     consecutive words, (3) it beats the expected reading by the
 *     pre-frozen slip margin, (4) the expected alignment has actually
 *     collapsed in that window. Otherwise the verdict is U (abstain)
 *     or a plain word-level difference with no ayah named.
 *
 * Documented limits of this implementation:
 * - The single-unit alternatives tested for a word are every token id
 *   of the model vocabulary except blank, the word bar and the
 *   expected unit itself (the full doc method: "every single-unit
 *   edit inside the word"). An alternative the decoder never produced
 *   IS tested, because the LLR — not the greedy text — is the
 *   evidence.
 * - Scoring runs once per take, after `session.finish()` returns the
 *   whole-take heads. Its latency is measured by the PoC (ayah
 *   completion), not assumed.
 * - Takes are scored per ayah (the PoC protocol). A take whose
 *   frames x extended-sequence length exceeds [MAX_CELLS] is refused
 *   with a clear error instead of risking an out-of-memory crash.
 * - The word-bar id defaults to the V4 vocabulary value (1) and the
 *   blank to 0 (verified from the 4.0.1 AAR's `v4_tokens.txt`); the
 *   caller SHOULD pass `tokens.wordBar` from the loaded model.
 */
enum class WordVerdict { C, S, D, H, U, I, R }

data class WordJudgement(
    val surah: Int,
    val ayah: Int,
    val wordIndex: Int,
    val verdict: WordVerdict,
    /** Fused LLR (log_probs + inter1_log_probs) when available; null if abstained. */
    val fusedLlr: Double? = null,
    /** For S / slip verdicts: the alternative ayah named, if any. Null otherwise. */
    val alternativeAyah: String? = null,
)

/**
 * Frame-level CTC log-probabilities of one finished take. Plain data —
 * no SDK types — so this file stays testable without a model.
 * `inter1LogProbs` MUST already be prepared as the calibration says:
 * the word bar merged into blank (see [prepareInter1LogProbs]).
 * `logProbs` is used exactly as the model emitted it.
 * `tokenNames[i]` is the vocabulary name of token id `i`.
 */
class CtcHeads(
    val logProbs: Array<FloatArray>,
    val inter1LogProbs: Array<FloatArray>,
    val classes: Int,
    val blankId: Int,
    val wordBarId: Int,
    val tokenNames: List<String>,
) {
    val frames: Int get() = logProbs.size
}

/**
 * Read-only access to the loaded model's `calibration.json`, exactly
 * as shipped. Implemented by the gateway from `model.calibration`
 * (see RealTajweedGateway). No threshold or weight value is ever
 * copied into code — every lookup goes through here.
 */
interface CalibrationSource {
    val contexts: List<String>
    val classes: List<String>
    val presets: List<String>
    val defaultContext: String
    val defaultPreset: String
    fun fusionWeight(context: String, klass: String): Double?
    fun threshold(context: String, klass: String, preset: String): Double?
    fun status(context: String, klass: String): String?
}

/** Everything the LLR alignment needs from one finished take. */
data class AlignmentEvidence(
    val heads: CtcHeads,
    val calibration: CalibrationSource,
    /** Calibration context, e.g. `calibration.defaultContext`. */
    val context: String,
    /** Threshold preset; the PoC uses "balanced" when the model offers it. */
    val preset: String,
)

/**
 * Prepares `inter1_log_probs` exactly as the calibration's
 * `inter1_prep` rule says: the word bar merged into blank. Returns a
 * new array; the input is not mutated. (AGENTS.md: "the `inter1_prep`
 * rule (the word bar merged into blank)".)
 */
internal fun prepareInter1LogProbs(
    raw: Array<FloatArray>,
    blankId: Int,
    wordBarId: Int,
): Array<FloatArray> {
    if (wordBarId < 0) return raw
    return Array(raw.size) { f ->
        val row = raw[f].copyOf()
        if (wordBarId < row.size && blankId < row.size) {
            row[blankId] = logSumExp2(
                row[blankId].toDouble(),
                row[wordBarId].toDouble(),
            ).toFloat()
            row[wordBarId] = Float.NEGATIVE_INFINITY
        }
        row
    }
}

/** log(exp(a) + exp(b)) in a numerically safe way. */
internal fun logSumExp2(a: Double, b: Double): Double {
    if (a == Double.NEGATIVE_INFINITY) return b
    if (b == Double.NEGATIVE_INFINITY) return a
    val m = max(a, b)
    return m + ln(exp(a - m) + exp(b - m))
}

private data class ExpectedWord(val index: Int, val units: List<Int>)

private enum class StepKind { PAIR, DELETE, INSERT, REPEAT }

private data class DpStep(val kind: StepKind, val expected: Int, val heard: Int)

class Alignment {

    fun align(
        expected: ExpectedAyah,
        heardTokenIds: List<Int>,
        acceptThreshold: Double,
        rejectThreshold: Double,
        slipMargin: Double,
        evidence: AlignmentEvidence? = null,
        /** label ("surah:ayah") -> that ayah's words' unit lists, for slip detection. */
        slipCandidates: Map<String, List<List<Int>>> = emptyMap(),
        /** Word-boundary token id; pass `tokens.wordBar` from the loaded model. */
        wordBarId: Int = DEFAULT_WORD_BAR_ID,
        /** Authoritative SDK word segmentation from decoded.words, if available. */
        heardWordTokenIds: List<List<Int>>? = null,
    ): List<WordJudgement> {
        require(acceptThreshold > rejectThreshold) { "acceptThreshold must exceed rejectThreshold" }
        require(slipMargin > 0) { "slipMargin must be positive (frozen before test set)" }
        val expectedWords = segmentExpected(expected)
        if (expectedWords.isEmpty()) return emptyList()
        // Margins and judgements are indexed by position in expectedWords;
        // the reference word indices must therefore be contiguous from 0.
        require(expectedWords.indices.all { expectedWords[it].index == it }) {
            "Expected word indices must be contiguous from 0 (got ${expectedWords.map { it.index }})"
        }
        val blankId = evidence?.heads?.blankId ?: DEFAULT_BLANK_ID

        // --- Heard words: prefer SDK-provided decoded.words. The SDK's
        // flat token stream is not guaranteed to contain word bars.
        // ---
        val heardWords = heardWordTokenIds?.map { word ->
            word.filter { it != blankId && it != wordBarId }
        }?.filter { it.isNotEmpty() }
            ?: segmentHeard(heardTokenIds, wordBarId, blankId)

        // Fail closed: a single decoded word cannot support word-level
        // judgments for a multi-word ayah. Previously this silently caused
        // DP to pair the lone word with the last expected word and label
        // all others D. Return U rather than confident false deletions.
        if (expectedWords.size > 1 && heardWords.size <= 1) {
            return expectedWords.map { w ->
                WordJudgement(expected.surah, expected.ayah, w.index, WordVerdict.U)
            }
        }

        // --- Restarts: split the heard stream into runs; the LAST run is
        // scored, earlier runs are repetitions (R). A split needs the
        // heard word to equal the first expected word AND the next heard
        // word to equal the second expected word, so a word the ayah
        // legitimately repeats later is never mistaken for a restart.
        // Assumption (PoC protocol): a take starts at the ayah's first word.
        val (scoredHeard, restartRepeats) =
            splitRestartRuns(expected, expectedWords, heardWords)

        // --- Word-level edit alignment (DP) over the scored run. ---
        val steps = alignWords(expectedWords, scoredHeard)
        val pairedHeard = IntArray(expectedWords.size) { -1 } // expected idx -> heard idx
        val heardDisposition = arrayOfNulls<StepKind>(scoredHeard.size)
        var lastPairedExpected = -1
        val repeatJudgements = mutableListOf<WordJudgement>()
        for (step in steps) {
            when (step.kind) {
                StepKind.PAIR -> {
                    pairedHeard[step.expected] = step.heard
                    heardDisposition[step.heard] = StepKind.PAIR
                    lastPairedExpected = step.expected
                }
                StepKind.DELETE -> { /* expected word has no heard word */ }
                StepKind.INSERT -> {
                    heardDisposition[step.heard] = StepKind.INSERT
                }
                StepKind.REPEAT -> {
                    heardDisposition[step.heard] = StepKind.REPEAT
                    repeatJudgements += WordJudgement(
                        expected.surah, expected.ayah,
                        if (lastPairedExpected >= 0) lastPairedExpected else 0,
                        WordVerdict.R,
                    )
                }
            }
        }

        val judgements = ArrayList<WordJudgement>(expectedWords.size)
        if (evidence == null) {
            // --- Token-only path: no LLR, so never accuse (no S). ---
            for (w in expectedWords) {
                val h = pairedHeard[w.index]
                val verdict = when {
                    h < 0 -> WordVerdict.D
                    scoredHeard[h] == w.units -> WordVerdict.C
                    else -> WordVerdict.U // differs, but unconfirmed without LLR
                }
                judgements += WordJudgement(expected.surah, expected.ayah, w.index, verdict)
            }
        } else if (evidence.heads.frames == 0) {
            // A take with no audio frames: abstain on every word. A
            // silent take is an invalid session, not a deletion — the
            // session must be re-run, not scored as D.
            for (w in expectedWords) {
                judgements += WordJudgement(expected.surah, expected.ayah, w.index, WordVerdict.U)
            }
        } else {
            // --- Heads path: the full LLR method. ---
            val scorer = LlrScorer(expected, expectedWords, evidence)
            val margins = DoubleArray(expectedWords.size) { Double.NEGATIVE_INFINITY }
            for (w in expectedWords) {
                val h = pairedHeard[w.index]
                val result = if (h < 0) {
                    scorer.deletionTest(w)
                } else {
                    scorer.substitutionTest(w, scoredHeard[h])
                }
                margins[w.index] = result.margin
                val verdict = when {
                    result.margin >= acceptThreshold -> result.verdict
                    result.margin <= rejectThreshold -> WordVerdict.C
                    else -> WordVerdict.U
                }
                // A deletion whose audio evidence is absent stays D only
                // when confirmed; a "C" here means the word IS in the
                // audio (the greedy decoder dropped it).
                judgements += WordJudgement(
                    expected.surah, expected.ayah, w.index, verdict,
                    fusedLlr = result.fusedLlr,
                )
            }
            // Inserted heard words (not repeats): confirm with LLR.
            for ((heardIdx, kind) in heardDisposition.withIndex()) {
                if (kind != StepKind.INSERT) continue
                val insertAt = insertionPoint(pairedHeard, heardIdx)
                val result = scorer.insertionTest(scoredHeard[heardIdx], insertAt)
                if (result.margin >= acceptThreshold) {
                    judgements += WordJudgement(
                        expected.surah, expected.ayah,
                        insertAt.coerceIn(0, expectedWords.size - 1),
                        WordVerdict.I, fusedLlr = result.fusedLlr,
                    )
                }
                // Unconfirmed insertions are NOT counted: an insertion the
                // audio does not support is a decoding artefact, not an
                // error. This is abstention, not a missed detection.
            }
            // Slip (mutashabihat) guard: all four conditions or nothing.
            applySlipGuard(
                expected, expectedWords, scoredHeard, pairedHeard,
                margins, judgements, scorer, slipCandidates,
                acceptThreshold, slipMargin,
            )
        }
        judgements += restartRepeats
        judgements += repeatJudgements
        return judgements
    }

    // ------------------------------------------------------------------
    // Segmentation and the word-level DP
    // ------------------------------------------------------------------

    private fun segmentExpected(expected: ExpectedAyah): List<ExpectedWord> {
        val words = ArrayList<ExpectedWord>()
        var i = 0
        while (i < expected.units.size) {
            val idx = expected.wordIndex[i]
            var j = i
            while (j < expected.units.size && expected.wordIndex[j] == idx) j++
            words += ExpectedWord(idx, expected.units.subList(i, j))
            i = j
        }
        return words
    }

    private fun segmentHeard(
        tokenIds: List<Int>,
        wordBarId: Int,
        blankId: Int,
    ): List<List<Int>> {
        val words = ArrayList<List<Int>>()
        var cur = ArrayList<Int>()
        for (id in tokenIds) {
            when (id) {
                wordBarId -> {
                    if (cur.isNotEmpty()) {
                        words += cur
                        cur = ArrayList()
                    }
                }
                blankId -> { /* decoded tokens carry no blanks; drop defensively */ }
                else -> cur += id
            }
        }
        if (cur.isNotEmpty()) words += cur
        return words
    }

    /**
     * Splits the heard words into runs at restart points. Returns the
     * last run (the scored one) and R judgements for the earlier runs'
     * words, whose expected index is their ordinal position in the
     * aborted run (a restart replays the ayah from its first word).
     */
    private fun splitRestartRuns(
        expected: ExpectedAyah,
        expectedWords: List<ExpectedWord>,
        heardWords: List<List<Int>>,
    ): Pair<List<List<Int>>, List<WordJudgement>> {
        var splitAt = -1
        val first = expectedWords[0].units
        val second = if (expectedWords.size > 1) expectedWords[1].units else null
        for (k in 1 until heardWords.size) {
            if (heardWords[k] != first) continue
            val continues = second == null ||
                (k + 1 < heardWords.size && heardWords[k + 1] == second)
            if (continues) splitAt = k // latest restart wins
        }
        if (splitAt < 0) return heardWords to emptyList()
        val repeats = ArrayList<WordJudgement>()
        for (r in 0 until splitAt) {
            repeats += WordJudgement(
                expected.surah, expected.ayah,
                minOf(r, expectedWords.size - 1), WordVerdict.R,
            )
        }
        return heardWords.subList(splitAt, heardWords.size) to repeats
    }

    private fun alignWords(
        expectedWords: List<ExpectedWord>,
        heardWords: List<List<Int>>,
    ): List<DpStep> {
        val n = expectedWords.size
        val m = heardWords.size
        val inf = Double.POSITIVE_INFINITY
        val cost = Array(n + 1) { DoubleArray(m + 1) { inf } }
        val back = Array(n + 1) { arrayOfNulls<DpStep>(m + 1) }
        cost[0][0] = 0.0
        for (i in 0..n) {
            for (j in 0..m) {
                val c = cost[i][j]
                if (c == inf) continue
                // Relax in priority order; strict < keeps the first (best)
                // transition on ties: PAIR, REPEAT, DELETE, INSERT.
                if (i < n && j < m) {
                    val pair = c + if (expectedWords[i].units == heardWords[j]) 0.0 else SUB_COST
                    if (pair < cost[i + 1][j + 1]) {
                        cost[i + 1][j + 1] = pair
                        back[i + 1][j + 1] = DpStep(StepKind.PAIR, i, j)
                    }
                }
                if (j < m && j >= 1 && heardWords[j] == heardWords[j - 1]) {
                    if (c + REPEAT_COST < cost[i][j + 1]) {
                        cost[i][j + 1] = c + REPEAT_COST
                        back[i][j + 1] = DpStep(StepKind.REPEAT, i, j)
                    }
                }
                if (i < n) {
                    if (c + DEL_COST < cost[i + 1][j]) {
                        cost[i + 1][j] = c + DEL_COST
                        back[i + 1][j] = DpStep(StepKind.DELETE, i, j)
                    }
                }
                if (j < m) {
                    if (c + INS_COST < cost[i][j + 1]) {
                        cost[i][j + 1] = c + INS_COST
                        back[i][j + 1] = DpStep(StepKind.INSERT, i, j)
                    }
                }
            }
        }
        // Trace back from (n, m).
        val steps = ArrayList<DpStep>()
        var i = n
        var j = m
        while (i > 0 || j > 0) {
            val step = back[i][j]
                ?: throw IllegalStateException("Alignment DP has no path to ($i, $j)")
            steps += step
            when (step.kind) {
                StepKind.PAIR -> { i--; j-- }
                StepKind.DELETE -> { i-- }
                StepKind.INSERT, StepKind.REPEAT -> { j-- }
            }
        }
        steps.reverse()
        return steps
    }

    /** Expected-word index at which an inserted heard word belongs. */
    private fun insertionPoint(pairedHeard: IntArray, heardIdx: Int): Int {
        // First expected word after the insertion point: the smallest
        // expected index whose paired heard index is greater than heardIdx.
        for (e in pairedHeard.indices) {
            if (pairedHeard[e] > heardIdx) return e
        }
        return pairedHeard.size // appended after the last expected word
    }

    // ------------------------------------------------------------------
    // Slip (mutashabihat) guard
    // ------------------------------------------------------------------

    private fun applySlipGuard(
        expected: ExpectedAyah,
        expectedWords: List<ExpectedWord>,
        scoredHeard: List<List<Int>>,
        pairedHeard: IntArray,
        margins: DoubleArray,
        judgements: ArrayList<WordJudgement>,
        scorer: LlrScorer,
        slipCandidates: Map<String, List<List<Int>>>,
        acceptThreshold: Double,
        slipMargin: Double,
    ) {
        if (slipCandidates.isEmpty()) return
        val marked = BooleanArray(expectedWords.size)
        for ((label, candWords) in slipCandidates) {
            for (start in expectedWords.indices) {
                if (marked[start]) continue
                for (k in candWords.indices) {
                    var len = 0
                    while (start + len < expectedWords.size &&
                        k + len < candWords.size
                    ) {
                        val h = pairedHeard[start + len]
                        if (h < 0 || scoredHeard[h] != candWords[k + len]) break
                        len++
                    }
                    if (len < 3) continue
                    val end = start + len - 1
                    // (1) every word in the window is above the accept
                    // threshold — no U word in the window.
                    if ((start..end).any { margins[it] < acceptThreshold }) continue
                    // (4) the expected alignment actually collapsed here:
                    // no confirmed-correct word in the window.
                    if ((start..end).any { judgements[it].verdict == WordVerdict.C }) continue
                    // (3) the alternative beats the expected reading by
                    // the pre-frozen margin, over the whole window.
                    val windowLlr = scorer.slipWindowLlr(start, end, candWords.subList(k, k + len))
                    if (windowLlr == null || windowLlr < slipMargin) continue
                    // (2) holds by construction (len >= 3). All four hold:
                    // name the alternative ayah on every word in the window.
                    for (w in start..end) {
                        marked[w] = true
                        val old = judgements[w]
                        judgements[w] = old.copy(
                            verdict = WordVerdict.S,
                            alternativeAyah = label,
                        )
                    }
                }
            }
        }
    }

    companion object {
        /** Word-boundary token id in the V4 vocabulary (verified from the AAR). */
        const val DEFAULT_WORD_BAR_ID: Int = 1

        /** CTC blank token id in the V4 vocabulary (verified from the AAR). */
        const val DEFAULT_BLANK_ID: Int = 0

        /**
         * Clip of the fused LLR, documented in AGENTS.md
         * ("the clip of the LLRs ([-30, 30])").
         */
        const val LLR_CLIP_MIN: Double = -30.0
        const val LLR_CLIP_MAX: Double = 30.0

        /** Largest single DP step costs (alignment-path costs only). */
        private const val SUB_COST: Double = 2.0
        private const val DEL_COST: Double = 2.0
        private const val INS_COST: Double = 2.0
        private const val REPEAT_COST: Double = 0.0

        /** Refuse takes whose forward-backward tables would exceed this many cells. */
        internal const val MAX_CELLS: Long = 2_000_000L

        /** Verbal accuracy C / (C+S+D+I); null = "insufficient evidence". */
        fun verbalAccuracy(c: Int, s: Int, d: Int, i: Int): Double? {
            val denom = c + s + d + i
            return if (denom == 0) null else c.toDouble() / denom
        }

        /** Independent recall C / (N-H-U); null = "insufficient evidence". */
        fun independentRecall(c: Int, n: Int, h: Int, u: Int): Double? {
            val denom = n - h - u
            return if (denom == 0) null else c.toDouble() / denom
        }

        /** Counts of each verdict in a judgement list. */
        fun counts(judgements: List<WordJudgement>): Map<WordVerdict, Int> =
            judgements.groupingBy { it.verdict }.eachCount()

        /** Worked example from the PoC (must reproduce exactly):
         *  N=100, C=80, S=6, D=5, H=6, U=3, I=4, R=7
         *  -> accuracy 80/95 = 84.2%, recall 80/91 = 87.9%. */
        fun workedExample(): Pair<Double, Double> {
            val acc = verbalAccuracy(80, 6, 5, 4)!!
            val rec = independentRecall(80, 100, 6, 3)!!
            check(kotlin.math.abs(acc - 0.842) < 0.001 && kotlin.math.abs(rec - 0.879) < 0.001) {
                "Worked example mismatch: $acc / $rec — definition has drifted from the PoC."
            }
            return acc to rec
        }
    }
}

// ----------------------------------------------------------------------
// LLR scorer (heads path) and the CTC core
// ----------------------------------------------------------------------

private data class LlrResult(
    /** Largest (fused LLR - class threshold) over the tested edits. */
    val margin: Double,
    val fusedLlr: Double?,
    /** The verdict this test would confirm (S, D or I). */
    val verdict: WordVerdict,
)

/**
 * The LLR method for one take. Built once per [Alignment.align] call;
 * the two forward-backward passes over the expected sequence are
 * computed lazily (only if some word actually needs them) and shared
 * by every single-unit alternative (the O(T) substitution trick).
 *
 * All thresholds and fusion weights come from [CalibrationSource]
 * (the loaded model's calibration.json) — nothing is hard-coded.
 */
private class LlrScorer(
    private val expected: ExpectedAyah,
    private val expectedWords: List<ExpectedWord>,
    private val evidence: AlignmentEvidence,
) {
    private val heads = evidence.heads
    private val calibration = evidence.calibration
    private val context = evidence.context
    private val preset = evidence.preset
    private val blankId = heads.blankId
    private val wordBarId = heads.wordBarId
    private val tokenNames = heads.tokenNames
    private val expectedUnits = expected.units

    /** Global unit offset of each expected word. */
    private val wordStart: IntArray = IntArray(expectedWords.size).also { offs ->
        var acc = 0
        for (i in expectedWords.indices) {
            offs[i] = acc
            acc += expectedWords[i].units.size
        }
    }

    private val fbLp: ForwardBackward by lazy {
        ForwardBackward(heads.logProbs, expectedUnits, blankId)
    }
    private val fbIlp: ForwardBackward by lazy {
        ForwardBackward(heads.inter1LogProbs, expectedUnits, blankId)
    }

    init {
        require(heads.frames > 0) { "No frames in the take's heads" }
        require(heads.logProbs.size == heads.inter1LogProbs.size) {
            "log_probs and inter1_log_probs have different frame counts"
        }
        require(heads.classes == tokenNames.size) {
            "tokenNames (${tokenNames.size}) does not match classes (${heads.classes})"
        }
        require(heads.logProbs.all { it.size == heads.classes }) {
            "A log_probs row does not have ${heads.classes} classes"
        }
        require(heads.inter1LogProbs.all { it.size == heads.classes }) {
            "An inter1_log_probs row does not have ${heads.classes} classes"
        }
        require(expectedUnits.none { it == blankId || it == wordBarId }) {
            "Expected units must not contain blank or word bar (strip them before scoring)"
        }
        val cells = heads.frames.toLong() * (2L * expectedUnits.size + 1)
        check(cells <= Alignment.MAX_CELLS) {
            "Take too long for full-sequence scoring ($cells cells); score per ayah."
        }
    }

    /** Largest clipped fused LLR over every single-unit edit inside the word. */
    fun substitutionTest(word: ExpectedWord, heardUnits: List<Int>): LlrResult {
        // heardUnits is not used for the evidence: the LLR is computed
        // from the heads, not from the greedy tokens. The greedy tokens
        // only decided which words are paired (the alignment above).
        var bestMargin = Double.NEGATIVE_INFINITY
        var bestFused: Double? = null
        val start = wordStart[word.index]
        for (p in word.units.indices) {
            val pos = start + p
            val expUnit = word.units[p]
            for (alt in 0 until heads.classes) {
                if (alt == blankId || alt == wordBarId || alt == expUnit) continue
                val klass = differenceClass(expUnit, alt)
                val (weight, threshold) = lookup(klass) ?: continue
                val llrLp = fbLp.substitutionLogProb(pos, alt) - fbLp.totalLogProb
                val llrIlp = fbIlp.substitutionLogProb(pos, alt) - fbIlp.totalLogProb
                val fused = fuse(llrLp, llrIlp, weight)
                val margin = fused - threshold
                if (margin > bestMargin) {
                    bestMargin = margin
                    bestFused = fused
                }
            }
        }
        return LlrResult(bestMargin, bestFused, WordVerdict.S)
    }

    /** Whole-word deletion plus every per-unit deletion inside the word. */
    fun deletionTest(word: ExpectedWord): LlrResult {
        var bestMargin = Double.NEGATIVE_INFINITY
        var bestFused: Double? = null
        fun consider(seq: List<Int>, klass: String) {
            val (weight, threshold) = lookup(klass) ?: return
            val llrLp = ctcForwardLogProb(heads.logProbs, seq, blankId) - fbLp.totalLogProb
            val llrIlp = ctcForwardLogProb(heads.inter1LogProbs, seq, blankId) - fbIlp.totalLogProb
            val fused = fuse(llrLp, llrIlp, weight)
            val margin = fused - threshold
            if (margin > bestMargin) {
                bestMargin = margin
                bestFused = fused
            }
        }
        val start = wordStart[word.index]
        val withoutWord = expectedUnits.toMutableList().apply {
            subList(start, start + word.units.size).clear()
        }
        // The whole-word deletion is judged against the class of the
        // word's first unit (a documented convention; the per-unit
        // deletions below carry each unit's own class).
        consider(withoutWord, deletionClass(word.units.first()))
        for (p in word.units.indices) {
            val withoutUnit = expectedUnits.toMutableList().apply { removeAt(start + p) }
            consider(withoutUnit, deletionClass(word.units[p]))
        }
        return LlrResult(bestMargin, bestFused, WordVerdict.D)
    }

    /** Whole-word insertion of the heard units at the given expected position. */
    fun insertionTest(units: List<Int>, insertAt: Int): LlrResult {
        val (weight, threshold) = lookup("insertion")
            ?: return LlrResult(Double.NEGATIVE_INFINITY, null, WordVerdict.I)
        val pos = if (insertAt >= expectedWords.size) expectedUnits.size
            else wordStart[insertAt]
        val withInsert = expectedUnits.toMutableList().apply { addAll(pos, units) }
        val llrLp = ctcForwardLogProb(heads.logProbs, withInsert, blankId) - fbLp.totalLogProb
        val llrIlp = ctcForwardLogProb(heads.inter1LogProbs, withInsert, blankId) - fbIlp.totalLogProb
        val fused = fuse(llrLp, llrIlp, weight)
        return LlrResult(fused - threshold, fused, WordVerdict.I)
    }

    /**
     * Fused LLR of the replacement sequence (expected words start..end
     * replaced by the candidate's words) against the expected sequence.
     * The fusion weight comes from the first differing unit's class; the
     * calibration threshold is NOT used here — the frozen [slipMargin]
     * governs the window (PoC). Null when the class is not judged or
     * the sequences do not differ.
     */
    fun slipWindowLlr(start: Int, end: Int, candWordUnits: List<List<Int>>): Double? {
        val from = wordStart[start]
        val to = wordStart[end] + expectedWords[end].units.size
        val replacement = expectedUnits.toMutableList().apply {
            subList(from, to).clear()
            addAll(from, candWordUnits.flatten())
        }
        val firstDiff = (0 until minOf(replacement.size, expectedUnits.size))
            .firstOrNull { replacement[it] != expectedUnits[it] } ?: return null
        val klass = differenceClass(expectedUnits[firstDiff], replacement[firstDiff])
        val (weight, _) = lookup(klass) ?: return null
        val llrLp = ctcForwardLogProb(heads.logProbs, replacement, blankId) - fbLp.totalLogProb
        val llrIlp = ctcForwardLogProb(heads.inter1LogProbs, replacement, blankId) - fbIlp.totalLogProb
        return fuse(llrLp, llrIlp, weight)
    }

    // -- calibration lookups -------------------------------------------------

    /** (fusion weight, threshold) for a class, or null if it is not judged. */
    private fun lookup(klass: String): Pair<Double, Double>? {
        // A class whose status is "off" is not judged (docs/migrating-to-3.md §3).
        if (calibration.status(context, klass) == "off") return null
        val weight = calibration.fusionWeight(context, klass) ?: return null
        val threshold = calibration.threshold(context, klass, preset) ?: return null
        return weight to threshold
    }

    private fun fuse(llrLp: Double, llrIlp: Double, weight: Double): Double =
        (weight * llrLp + (1.0 - weight) * llrIlp)
            .coerceIn(Alignment.LLR_CLIP_MIN, Alignment.LLR_CLIP_MAX)

    // -- unit classification (V4 vocabulary names, verified from v4_tokens.txt) --

    private fun isShortVowel(name: String): Boolean =
        name == "a" || name == "i" || name == "u"

    private fun isLongVowel(name: String): Boolean =
        name == "aa" || name == "ii" || name == "uu"

    private fun isVowel(name: String): Boolean = isShortVowel(name) || isLongVowel(name)

    private fun differenceClass(expUnit: Int, altUnit: Int): String {
        val expName = tokenNames[expUnit]
        val altName = tokenNames[altUnit]
        return when {
            isShortVowel(expName) && isShortVowel(altName) -> "vowel_sub"
            isShortVowel(expName) && isLongVowel(altName) -> "short_to_long"
            isLongVowel(expName) && isShortVowel(altName) -> "long_to_short"
            isLongVowel(expName) && isLongVowel(altName) -> "vowel_sub"
            !isVowel(expName) && !isVowel(altName) -> "cons_sub"
            // Cross-type (vowel <-> consonant): by the expected unit's type.
            isVowel(expName) -> "vowel_sub"
            else -> "cons_sub"
        }
    }

    private fun deletionClass(unit: Int): String =
        if (isVowel(tokenNames[unit])) "vowel_del" else "cons_del"
}

/** Expected sequence with blanks interleaved: [b, l0, b, l1, b, ...]. */
private fun extended(labels: List<Int>, blankId: Int): List<Int> {
    val ext = ArrayList<Int>(2 * labels.size + 1)
    ext += blankId
    for (label in labels) {
        ext += label
        ext += blankId
    }
    return ext
}

/** Full-sequence CTC forward log-probability, log space. */
private fun ctcForwardLogProb(
    logProbs: Array<FloatArray>,
    labels: List<Int>,
    blankId: Int,
): Double {
    val t = logProbs.size
    val u = labels.size
    if (u == 0) {
        var s = 0.0
        for (f in 0 until t) s += logProbs[f][blankId].toDouble()
        return s
    }
    if (t == 0) return Double.NEGATIVE_INFINITY
    val ext = extended(labels, blankId)
    val sLen = ext.size
    var prev = DoubleArray(sLen) { Double.NEGATIVE_INFINITY }
    prev[0] = logProbs[0][blankId].toDouble()
    prev[1] = logProbs[0][labels[0]].toDouble()
    for (frame in 1 until t) {
        val cur = DoubleArray(sLen) { Double.NEGATIVE_INFINITY }
        for (s in 0 until sLen) {
            var best = prev[s]
            if (s >= 1) best = logSumExp2(best, prev[s - 1])
            if (s >= 2 && ext[s] != blankId && ext[s] != ext[s - 2]) {
                best = logSumExp2(best, prev[s - 2])
            }
            cur[s] = logProbs[frame][ext[s]].toDouble() + best
        }
        prev = cur
    }
    return logSumExp2(prev[sLen - 1], prev[sLen - 2])
}

/**
 * Forward-backward tables for one head and the expected sequence.
 * [substitutionLogProb] returns log P of the sequence with
 * `labels[pos]` replaced by `newLabel`, in O(T) using the enter/beta
 * decomposition — validated against a fresh full forward in the
 * Python prototype (/tmp/ctc_prototype.py, all tests passed):
 * the suffix beta must be recomputed for the changed state, because
 * paths staying in that state emit the NEW label, and the skip
 * conditions must be re-checked against the new label.
 */
private class ForwardBackward(
    private val logProbs: Array<FloatArray>,
    labels: List<Int>,
    private val blankId: Int,
) {
    private val ext: List<Int> = extended(labels, blankId)
    private val frames: Int = logProbs.size
    private val alpha: Array<DoubleArray>
    private val beta: Array<DoubleArray>
    val totalLogProb: Double

    init {
        val sLen = ext.size
        alpha = Array(frames) { DoubleArray(sLen) { Double.NEGATIVE_INFINITY } }
        beta = Array(frames) { DoubleArray(sLen) { Double.NEGATIVE_INFINITY } }
        alpha[0][0] = logProbs[0][blankId].toDouble()
        if (sLen > 1) alpha[0][1] = logProbs[0][ext[1]].toDouble()
        for (t in 1 until frames) {
            for (s in 0 until sLen) {
                var v = alpha[t - 1][s]
                if (s >= 1) v = logSumExp2(v, alpha[t - 1][s - 1])
                if (s >= 2 && ext[s] != blankId && ext[s] != ext[s - 2]) {
                    v = logSumExp2(v, alpha[t - 1][s - 2])
                }
                alpha[t][s] = logProbs[t][ext[s]].toDouble() + v
            }
        }
        totalLogProb = if (sLen == 1) alpha[frames - 1][0]
            else logSumExp2(alpha[frames - 1][sLen - 1], alpha[frames - 1][sLen - 2])
        beta[frames - 1][sLen - 1] = 0.0
        if (sLen > 1) beta[frames - 1][sLen - 2] = 0.0
        for (t in frames - 2 downTo 0) {
            for (s in 0 until sLen) {
                var v = logProbs[t + 1][ext[s]].toDouble() + beta[t + 1][s]
                if (s + 1 < sLen) {
                    v = logSumExp2(v, logProbs[t + 1][ext[s + 1]].toDouble() + beta[t + 1][s + 1])
                }
                if (s + 2 < sLen && ext[s + 2] != blankId && ext[s + 2] != ext[s]) {
                    v = logSumExp2(v, logProbs[t + 1][ext[s + 2]].toDouble() + beta[t + 1][s + 2])
                }
                beta[t][s] = v
            }
        }
    }

    /** log P(labels with labels[pos] replaced by newLabel). */
    fun substitutionLogProb(pos: Int, newLabel: Int): Double {
        val s = 2 * pos + 1
        val sLen = ext.size
        val t = frames
        // beta_new: suffix values for the changed state s only, recomputed
        // because paths staying in s now emit the NEW label.
        val betaNew = DoubleArray(t) { Double.NEGATIVE_INFINITY }
        if (s == sLen - 1 || s == sLen - 2) betaNew[t - 1] = 0.0
        val skipOk = s + 2 < sLen && ext[s + 2] != blankId && ext[s + 2] != newLabel
        for (frame in t - 2 downTo 0) {
            var v = logProbs[frame + 1][newLabel].toDouble() + betaNew[frame + 1]
            if (s + 1 < sLen) {
                v = logSumExp2(v, logProbs[frame + 1][ext[s + 1]].toDouble() + beta[frame + 1][s + 1])
            }
            if (skipOk) {
                v = logSumExp2(v, logProbs[frame + 1][ext[s + 2]].toDouble() + beta[frame + 1][s + 2])
            }
            betaNew[frame] = v
        }
        var total = Double.NEGATIVE_INFINITY
        for (frame in 0 until t) {
            val arr = if (frame == 0) {
                if (s == 1) 0.0 else Double.NEGATIVE_INFINITY
            } else {
                var a = alpha[frame - 1][s - 1]
                if (s >= 2 && newLabel != blankId && newLabel != ext[s - 2]) {
                    a = logSumExp2(a, alpha[frame - 1][s - 2])
                }
                a
            }
            if (arr == Double.NEGATIVE_INFINITY) continue
            total = logSumExp2(total, arr + logProbs[frame][newLabel].toDouble() + betaNew[frame])
        }
        return total
    }
}
