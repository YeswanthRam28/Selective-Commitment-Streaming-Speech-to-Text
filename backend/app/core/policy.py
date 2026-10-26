import time
from typing import List, Tuple
from app.models.events import TranscriptSpan, SpanState, PolicyConfig, PolicyMode

def evaluate_commitment_policy(
    spans: List[TranscriptSpan],
    config: PolicyConfig,
    is_final: bool = False
) -> Tuple[List[TranscriptSpan], List[str]]:
    """
    Applies the configured policy (Selective Commitment, Raw Streaming, or Fixed Delay)
    to decide which spans transition to COMMITTED or FINALIZED.
    Returns: (updated_spans, newly_committed_ids)
    """
    now = time.time()
    updated_spans: List[TranscriptSpan] = []
    newly_committed_ids: List[str] = []

    # Finalization rule: at end-of-utterance, reconcile all spans
    if is_final:
        for span in spans:
            s = span.model_copy()
            if s.state != SpanState.FINALIZED:
                if s.state != SpanState.COMMITTED:
                    newly_committed_ids.append(s.span_id)
                s.state = SpanState.FINALIZED
                if not s.committed_at:
                    s.committed_at = now
            updated_spans.append(s)
        return updated_spans, newly_committed_ids

    if config.mode == PolicyMode.RAW_STREAMING:
        # Raw streaming does not commit anything before finalization
        return spans, []

    elif config.mode == PolicyMode.FIXED_DELAY:
        # Fixed delay: commit all spans that are at least 'fixed_delay_updates' behind the frontier
        total = len(spans)
        commit_cutoff = max(0, total - config.fixed_delay_updates)
        for idx, span in enumerate(spans):
            s = span.model_copy()
            if idx < commit_cutoff and s.state not in (SpanState.COMMITTED, SpanState.FINALIZED):
                s.state = SpanState.COMMITTED
                s.committed_at = now
                newly_committed_ids.append(s.span_id)
            updated_spans.append(s)
        return updated_spans, newly_committed_ids

    else:
        # SELECTIVE_COMMITMENT policy
        for span in spans:
            s = span.model_copy()
            if s.state in (SpanState.COMMITTED, SpanState.FINALIZED):
                updated_spans.append(s)
                continue

            # Rule 1: Hypothesis Stability (update count agreement)
            stable_by_updates = s.update_count >= config.required_stability_updates

            # Rule 2: Confidence threshold (if set and available)
            confident = True
            if config.confidence_threshold is not None and s.confidence is not None:
                confident = s.confidence >= config.confidence_threshold

            # Rule 3: Commitment deadline fallback
            expired_deadline = (now - s.first_displayed_at) >= config.max_provisional_duration_sec

            if (stable_by_updates and confident) or expired_deadline:
                s.state = SpanState.COMMITTED
                s.committed_at = now
                newly_committed_ids.append(s.span_id)

            updated_spans.append(s)

        return updated_spans, newly_committed_ids
