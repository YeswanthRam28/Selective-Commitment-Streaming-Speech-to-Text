import time
from typing import List, Optional, Tuple
from app.models.events import (
    TranscriptSpan, SpanState, PolicyConfig, HypothesisEvent, TranscriptUpdateEvent
)
from app.core.alignment import align_spans_with_hypothesis
from app.core.policy import evaluate_commitment_policy

class TranscriptionSession:
    def __init__(self, session_id: str, config: PolicyConfig):
        self.session_id = session_id
        self.config = config
        self.spans: List[TranscriptSpan] = []
        self.event_seq = 0
        self.total_revisions = 0
        self.total_committed = 0
        self.created_at = time.time()
        self.first_word_display_time: Optional[float] = None
        self.latencies: List[float] = []

    def handle_hypothesis(self, hypothesis: HypothesisEvent) -> TranscriptUpdateEvent:
        self.event_seq += 1

        # 1. Alignment
        aligned_spans, revised_ids = align_spans_with_hypothesis(self.spans, hypothesis)
        self.total_revisions += len(revised_ids)

        if aligned_spans and self.first_word_display_time is None:
            self.first_word_display_time = time.time() - self.created_at

        # 2. Selective Commitment Policy
        updated_spans, newly_committed_ids = evaluate_commitment_policy(
            aligned_spans, self.config, is_final=hypothesis.is_final
        )
        self.total_committed += len(newly_committed_ids)
        self.spans = updated_spans

        # Compute full displayed text
        full_text = " ".join(s.text for s in self.spans)

        return TranscriptUpdateEvent(
            session_id=self.session_id,
            seq=self.event_seq,
            timestamp=time.time(),
            spans=self.spans,
            newly_committed_ids=newly_committed_ids,
            revised_ids=revised_ids,
            full_text=full_text,
            is_final=hypothesis.is_final
        )
