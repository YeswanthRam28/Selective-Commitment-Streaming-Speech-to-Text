import difflib
import uuid
import time
from typing import List, Tuple
from app.models.events import TranscriptSpan, SpanState, WordItem, HypothesisEvent

def align_spans_with_hypothesis(
    existing_spans: List[TranscriptSpan],
    hypothesis: HypothesisEvent
) -> Tuple[List[TranscriptSpan], List[str]]:
    """
    Aligns the current list of spans with the incoming hypothesis.
    - Preserves committed spans.
    - Updates, revises, or appends provisional spans.
    Returns: (updated_spans, revised_span_ids)
    """
    committed_spans = [s for s in existing_spans if s.state in (SpanState.COMMITTED, SpanState.FINALIZED)]
    provisional_spans = [s for s in existing_spans if s.state not in (SpanState.COMMITTED, SpanState.FINALIZED)]

    hyp_words = hypothesis.words
    revised_ids: List[str] = []

    # If hypothesis is empty, return existing committed spans
    if not hyp_words:
        return committed_spans, []

    # Map words to text tokens (strip whitespace/punct for matching comparison)
    hyp_tokens = [w.word.strip().lower() for w in hyp_words]
    committed_tokens = [s.text.strip().lower() for s in committed_spans]

    # Find where the committed prefix ends in the hypothesis
    start_hyp_idx = 0
    if committed_tokens:
        matcher = difflib.SequenceMatcher(None, committed_tokens, hyp_tokens)
        match = matcher.find_longest_match(0, len(committed_tokens), 0, len(hyp_tokens))
        if match.a == 0 and match.size > 0:
            start_hyp_idx = match.b + match.size
        else:
            # Fallback: advance start_hyp_idx by matching committed length if within bounds
            start_hyp_idx = min(len(committed_tokens), len(hyp_words))

    uncommitted_hyp_words = hyp_words[start_hyp_idx:]
    if not uncommitted_hyp_words:
        return committed_spans, []

    prev_prov_tokens = [s.text.strip().lower() for s in provisional_spans]
    new_prov_tokens = [w.word.strip().lower() for w in uncommitted_hyp_words]

    # Match previous provisional spans with new provisional tokens
    matcher = difflib.SequenceMatcher(None, prev_prov_tokens, new_prov_tokens)
    opcodes = matcher.get_opcodes()

    new_provisional_spans: List[TranscriptSpan] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Tokens are identical across updates
            for prev_idx, new_idx in zip(range(i1, i2), range(j1, j2)):
                old_span = provisional_spans[prev_idx]
                w = uncommitted_hyp_words[new_idx]
                updated_span = old_span.model_copy()
                updated_span.text = w.word
                updated_span.update_count += 1
                updated_span.confidence = w.confidence
                updated_span.start_time = w.start_time or old_span.start_time
                updated_span.end_time = w.end_time or old_span.end_time
                new_provisional_spans.append(updated_span)

        elif tag == 'replace':
            # Substitution / revision
            for prev_idx, new_idx in zip(range(i1, i2), range(j1, j2)):
                old_span = provisional_spans[prev_idx]
                w = uncommitted_hyp_words[new_idx]
                revised_span = old_span.model_copy()
                if old_span.text != w.word:
                    revised_span.history.append(old_span.text)
                    revised_span.revised_count += 1
                    revised_span.state = SpanState.REVISED
                    revised_ids.append(revised_span.span_id)
                revised_span.text = w.word
                revised_span.update_count = 1
                revised_span.confidence = w.confidence
                revised_span.start_time = w.start_time or old_span.start_time
                revised_span.end_time = w.end_time or old_span.end_time
                new_provisional_spans.append(revised_span)

            # Extra new tokens in replacement
            if (j2 - j1) > (i2 - i1):
                for new_idx in range(j1 + (i2 - i1), j2):
                    w = uncommitted_hyp_words[new_idx]
                    new_span = TranscriptSpan(
                        span_id=f"span_{uuid.uuid4().hex[:8]}",
                        text=w.word,
                        state=SpanState.PROVISIONAL,
                        start_time=w.start_time,
                        end_time=w.end_time,
                        confidence=w.confidence,
                        update_count=1,
                        first_displayed_at=time.time()
                    )
                    new_provisional_spans.append(new_span)

        elif tag == 'insert':
            for new_idx in range(j1, j2):
                w = uncommitted_hyp_words[new_idx]
                new_span = TranscriptSpan(
                    span_id=f"span_{uuid.uuid4().hex[:8]}",
                    text=w.word,
                    state=SpanState.PROVISIONAL,
                    start_time=w.start_time,
                    end_time=w.end_time,
                    confidence=w.confidence,
                    update_count=1,
                    first_displayed_at=time.time()
                )
                new_provisional_spans.append(new_span)

        elif tag == 'delete':
            # Word was deleted in newer hypothesis - omitted from provisional spans
            pass

    return committed_spans + new_provisional_spans, revised_ids
