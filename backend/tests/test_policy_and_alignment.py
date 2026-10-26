import pytest
from app.models.events import (
    TranscriptSpan, SpanState, PolicyConfig, PolicyMode, HypothesisEvent, WordItem
)
from app.core.alignment import align_spans_with_hypothesis
from app.core.policy import evaluate_commitment_policy
from app.metrics.evaluator import compute_wer, normalize_text

def test_normalize_text():
    assert normalize_text("Hello, World! Here is 123.") == "hello world here is 123"

def test_compute_wer_exact_match():
    res = compute_wer("this is a test", "this is a test")
    assert res["wer"] == 0.0

def test_compute_wer_substitution_and_deletion():
    res = compute_wer("the quick brown fox", "the fast brown")
    # ref length = 4. Edits: 1 sub (quick->fast), 1 del (fox) => 2/4 = 0.5
    assert res["wer"] == 0.5

def test_alignment_incremental_and_revision():
    spans = []

    # Update 1: "We are test"
    hyp1 = HypothesisEvent(
        session_id="s1", seq=1, is_final=False,
        words=[WordItem(word="We"), WordItem(word="are"), WordItem(word="test")],
        raw_text="We are test"
    )
    spans, revised_ids = align_spans_with_hypothesis(spans, hyp1)
    assert len(spans) == 3
    assert spans[0].text == "We"
    assert spans[2].text == "test"
    assert len(revised_ids) == 0

    # Update 2: "We are testing" (revision of 'test' to 'testing')
    hyp2 = HypothesisEvent(
        session_id="s1", seq=2, is_final=False,
        words=[WordItem(word="We"), WordItem(word="are"), WordItem(word="testing")],
        raw_text="We are testing"
    )
    spans, revised_ids = align_spans_with_hypothesis(spans, hyp2)
    assert len(spans) == 3
    assert spans[0].update_count == 2
    assert spans[2].text == "testing"
    assert spans[2].state == SpanState.REVISED
    assert spans[2].revised_count == 1
    assert "test" in spans[2].history
    assert len(revised_ids) == 1

def test_selective_commitment_policy():
    config = PolicyConfig(
        mode=PolicyMode.SELECTIVE_COMMITMENT,
        required_stability_updates=2
    )

    span1 = TranscriptSpan(span_id="s1", text="We", update_count=1)
    span2 = TranscriptSpan(span_id="s2", text="are", update_count=2)

    # span1 has update_count=1 (< 2), so remains provisional
    # span2 has update_count=2 (>= 2), so should commit
    updated, newly_committed = evaluate_commitment_policy([span1, span2], config)

    assert updated[0].state == SpanState.PROVISIONAL
    assert updated[1].state == SpanState.COMMITTED
    assert "s2" in newly_committed
    assert "s1" not in newly_committed

def test_raw_streaming_baseline():
    config = PolicyConfig(mode=PolicyMode.RAW_STREAMING)
    span = TranscriptSpan(span_id="s1", text="We", update_count=10)
    updated, newly_committed = evaluate_commitment_policy([span], config, is_final=False)

    # Raw streaming keeps it provisional before finalization
    assert updated[0].state == SpanState.PROVISIONAL
    assert len(newly_committed) == 0

    # Once is_final is True, it finalizes
    finalized, committed_ids = evaluate_commitment_policy([span], config, is_final=True)
    assert finalized[0].state == SpanState.FINALIZED
    assert "s1" in committed_ids
