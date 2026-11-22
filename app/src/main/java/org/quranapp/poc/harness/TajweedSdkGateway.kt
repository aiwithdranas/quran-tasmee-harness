package org.quranapp.poc.harness

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emptyFlow

/**
 * Gateway to the Quran Lab SDK — interface + STUB.
 *
 * The real implementation (one file, `RealTajweedGateway.kt`, to be added
 * ONLY after SDK access is granted) does exactly the four calls from the
 * Android quick start:
 *   TajweedClient(context, key) -> client.model() -> model.listen(...)
 *   -> session.chunks (chunk.decoded: text, tokens, words, settled, isFinal;
 *      chunk.heads: log_probs, inter1_log_probs, timing_log_probs, gem,
 *      attr_qalqalah, conf, …)
 * and `model.session()` + `session.push(samples)` for replaying saved WAVs.
 *
 * The STUB below emits nothing and records that it is a stub. It exists
 * so the Harness compiles and its UI/alignment code can be tested without
 * a key, a model download, or a phone. It MUST be replaced before any
 * live session; a live session attempted through the stub fails loudly
 * instead of faking results.
 */
data class HeardChunk(
    val summary: String,
    /** Token ids heard so far, in model.tokens (38 tokens). Empty in stub. */
    val tokenIds: List<Int> = emptyList(),
    /** Per-token confidence, parallel to tokenIds. Empty in stub. */
    val confidences: List<Double> = emptyList(),
    val isFinal: Boolean = false,
)

interface TajweedSdkGateway {
    fun startSession(): Flow<HeardChunk>
    val isStub: Boolean
}

class StubTajweedGateway : TajweedSdkGateway {
    override val isStub: Boolean = true
    override fun startSession(): Flow<HeardChunk> {
        // STUB: no model, no microphone, no recognition.
        // Deliberately returns an empty flow; MainActivity logs the stub state.
        return emptyFlow()
    }
}
