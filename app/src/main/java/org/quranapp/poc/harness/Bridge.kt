package org.quranapp.poc.harness

import android.content.Context
import org.json.JSONArray

/**
 * Bridge: expected phoneme sequence for an ayah.
 *
 * Source of truth: `model.reference` (unit ids + word boundaries per ayah),
 * joined to the phones/rules of the quran-tajweed-phonetics dataset on the
 * SETUP computer (see ../step0-findings.md). The Harness NEVER runs
 * quran-g2p (Python) on the phone. It reads a frozen, SHA-256-checked JSON
 * file bundled in assets: `reference_67_32.json`.
 *
 * File format (one entry per ayah, 60 entries total):
 *   { "surah": 67, "ayah": 1, "text": "<Uthmani text>",
 *     "units": [3, 32, 13, ...],            // model.reference unit ids
 *     "wordIndex": [0, 0, 0, 1, 1, ...] }   // word index per unit
 *
 * STATUS: the JSON file EXISTS (assets/reference_67_32.json, 60 ayahs,
 * 4,146 units, SHA-256 ce080dbf…) and is verified against
 * `model.reference` at every launch (see MainActivity): a short vowel
 * + its madd letter is ONE long-vowel unit, exactly as quran-g2p's
 * tokenlayer_tj3.py `_emit` does (fatha+alef_madd -> aa, kasra+yeh_madd
 * -> ii, damma+waw_madd -> uu). The first generator version missed
 * that merge (4,578 units) and the launch check caught the mismatch
 * on all 60 ayahs on 2026-11-21; the generator was fixed
 * (432 madd merges) and the counts now match the model exactly.
 * The file stays PROVISIONAL until the launch check reports 60/60.
 * A custom phoneme bridge is NOT built: if model.reference cannot
 * produce a word-mapped sequence for any of the 60 ayahs, that is a
 * Step 0 failure to report, not something to work around.
 */
data class ExpectedAyah(
    val surah: Int,
    val ayah: Int,
    val text: String,
    /** Unit ids from model.reference, in order, excluding word boundaries. */
    val units: List<Int>,
    /** Word index for each unit; same length as [units]. */
    val wordIndex: List<Int>,
)

interface Bridge {
    fun expectedAyah(surah: Int, ayah: Int): ExpectedAyah
}

class ReferenceBridge(
    /** Parsed from assets/reference_67_32.json; injected, never hardcoded. */
    private val byKey: Map<Pair<Int, Int>, ExpectedAyah>,
) : Bridge {
    override fun expectedAyah(surah: Int, ayah: Int): ExpectedAyah =
        byKey[Pair(surah, ayah)]
            ?: throw IllegalStateException(
                "STUB: reference_67_32.json not bundled yet — " +
                    "expected units for $surah:$ayah are not available. " +
                    "Generate it on the setup computer (see step0-findings.md)."
            )

    /** All bundled ayahs (for the model.reference verification). */
    fun all(): Collection<ExpectedAyah> = byKey.values
}

/** Loads the bundled reference JSON into a [ReferenceBridge]. */
object ReferenceLoader {
    fun loadFromAssets(context: Context): ReferenceBridge {
        val text = context.assets.open("reference_67_32.json")
            .bufferedReader().use { it.readText() }
        return loadFromJson(text)
    }

    fun loadFromJson(json: String): ReferenceBridge {
        val arr = JSONArray(json)
        val map = HashMap<Pair<Int, Int>, ExpectedAyah>()
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val units = o.getJSONArray("units")
                .let { a -> List(a.length()) { a.getInt(it) } }
            val wordIndex = o.getJSONArray("wordIndex")
                .let { a -> List(a.length()) { a.getInt(it) } }
            val ayah = ExpectedAyah(
                o.getInt("surah"), o.getInt("ayah"), o.getString("text"),
                units, wordIndex,
            )
            map[Pair(ayah.surah, ayah.ayah)] = ayah
        }
        return ReferenceBridge(map)
    }
}

data class ReferenceCheck(
    val total: Int,
    val matched: Int,
    val mismatches: List<String>,
)

/**
 * Compares the bundled JSON, unit-for-unit, against the loaded model's
 * own `model.reference`. The model's array carries the word-bar token
 * between words; it is stripped (leading/trailing bars too) before the
 * comparison, and the word boundaries it implies are checked against
 * the JSON's wordIndex. Any mismatch means the JSON is NOT the model's
 * reference and no session may be scored against it.
 */
suspend fun verifyReferenceAgainstModel(
    bridge: ReferenceBridge,
    unitsFor: suspend (surah: Int, ayah: Int) -> IntArray,
    wordBarId: Int,
): ReferenceCheck {
    val mismatches = ArrayList<String>()
    var matched = 0
    var total = 0
    for (ayah in bridge.all()) {
        total++
        val modelUnits = unitsFor(ayah.surah, ayah.ayah)
            .dropWhile { it == wordBarId }
            .dropLastWhile { it == wordBarId }
        val stripped = ArrayList<Int>()
        val derivedWordIndex = ArrayList<Int>()
        var w = 0
        for (u in modelUnits) {
            if (u == wordBarId) w++ else {
                stripped += u
                derivedWordIndex += w
            }
        }
        when {
            stripped != ayah.units ->
                mismatches += "${ayah.surah}:${ayah.ayah} units differ " +
                    "(json ${ayah.units.size} vs model ${stripped.size})"
            derivedWordIndex != ayah.wordIndex ->
                mismatches += "${ayah.surah}:${ayah.ayah} wordIndex differs"
            else -> matched++
        }
    }
    return ReferenceCheck(total, matched, mismatches)
}

/**
 * Slip (mutashabihat) candidates: the similar ayahs a reciter may slip
 * into, as unit sequences grouped by word — the shape [Alignment.align]
 * takes as `slipCandidates` (label "surah:ayah" -> that ayah's words'
 * unit lists). Source: the bundled `assets/slip_candidates.json`
 * (18 ayahs, generated by tools/generate_reference.py --mode slip
 * from the same phonetics dataset as the main reference; SHA-256
 * 8fb962d7…). It covers all 14 approved slip targets (4 tuning +
 * 10 test, including the three replacements approved 2026-11-21:
 * 16:78, 67:19, 32:28); the five extra ayahs in the file (10:48,
 * 23:78, 23:79, 32:27, 67:20) are harmless extra candidates —
 * 23:78 and 10:48 are the superseded X02/X04 targets, kept as
 * candidates because they are genuinely similar ayahs.
 *
 * Loading is separate from verification: the caller MUST verify the
 * candidates against the loaded model's own `model.reference` (build a
 * [ReferenceBridge] from [SlipCandidate.ayah] and run
 * [verifyReferenceAgainstModel]) before passing them to
 * [Alignment.align] — on any mismatch the candidates are dropped and
 * slip detection stays off, while ordinary word scoring still works.
 * Wired into MainActivity 2026-11-22 (launch check + align call).
 */
data class SlipCandidate(
    val ayah: ExpectedAyah,
    /** That ayah's units grouped by word, in order. */
    val words: List<List<Int>>,
)

object SlipCandidatesLoader {
    fun loadFromAssets(context: Context): Map<String, SlipCandidate> {
        val text = context.assets.open("slip_candidates.json")
            .bufferedReader().use { it.readText() }
        return loadFromJson(text)
    }

    fun loadFromJson(json: String): Map<String, SlipCandidate> {
        val arr = JSONArray(json)
        val map = HashMap<String, SlipCandidate>()
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val units = o.getJSONArray("units")
                .let { a -> List(a.length()) { a.getInt(it) } }
            val wordIndex = o.getJSONArray("wordIndex")
                .let { a -> List(a.length()) { a.getInt(it) } }
            val ayah = ExpectedAyah(
                o.getInt("surah"), o.getInt("ayah"), o.getString("text"),
                units, wordIndex,
            )
            map["${ayah.surah}:${ayah.ayah}"] = SlipCandidate(ayah, wordsOf(ayah))
        }
        return map
    }

    /** Groups an ayah's units by wordIndex, in order. */
    fun wordsOf(ayah: ExpectedAyah): List<List<Int>> {
        val words = ArrayList<List<Int>>()
        var cur = ArrayList<Int>()
        var curIdx = -1
        for (j in ayah.units.indices) {
            if (ayah.wordIndex[j] != curIdx) {
                if (cur.isNotEmpty()) words += cur
                cur = ArrayList()
                curIdx = ayah.wordIndex[j]
            }
            cur += ayah.units[j]
        }
        if (cur.isNotEmpty()) words += cur
        return words
    }
}
