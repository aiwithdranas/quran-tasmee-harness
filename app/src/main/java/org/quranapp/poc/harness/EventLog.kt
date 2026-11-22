package org.quranapp.poc.harness

import java.time.Instant

/**
 * Raw event log. Every session event is appended with a timestamp and
 * kept in memory + written to a local file by the caller. Scores are
 * NEVER stored here as final numbers — they are recomputed from this
 * log, so any verdict can be challenged line by line.
 * Log format: "<ISO-8601 UTC> <event> <detail>"
 * Recorded audio files (WAV, mono, 16 kHz) are stored separately on the
 * phone and referenced by SHA-256 in the log; no audio is sent anywhere.
 */
class EventLog {
    private val _lines = mutableListOf<String>()
    fun lines(): List<String> = _lines.toList()
    fun append(event: String, detail: String = "") {
        _lines += "${Instant.now()} $event ${detail}".trimEnd()
    }
}
