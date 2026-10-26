from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
import time

class SpanState(str, Enum):
    PROVISIONAL = "provisional"
    COMMITTED = "committed"
    REVISED = "revised"
    FINALIZED = "finalized"

class PolicyMode(str, Enum):
    RAW_STREAMING = "raw_streaming"
    FIXED_DELAY = "fixed_delay"
    SELECTIVE_COMMITMENT = "selective_commitment"

class WordItem(BaseModel):
    word: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: Optional[float] = None

class HypothesisEvent(BaseModel):
    session_id: str
    seq: int
    is_final: bool = False
    words: List[WordItem]
    raw_text: str
    timestamp: float = Field(default_factory=time.time)

class TranscriptSpan(BaseModel):
    span_id: str
    text: str
    state: SpanState = SpanState.PROVISIONAL
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: Optional[float] = None
    update_count: int = 1
    first_displayed_at: float = Field(default_factory=time.time)
    committed_at: Optional[float] = None
    revised_count: int = 0
    history: List[str] = Field(default_factory=list)

class TranscriptUpdateEvent(BaseModel):
    session_id: str
    seq: int
    timestamp: float = Field(default_factory=time.time)
    spans: List[TranscriptSpan]
    newly_committed_ids: List[str] = Field(default_factory=list)
    revised_ids: List[str] = Field(default_factory=list)
    full_text: str
    is_final: bool = False

class PolicyConfig(BaseModel):
    mode: PolicyMode = PolicyMode.SELECTIVE_COMMITMENT
    required_stability_updates: int = 2
    confidence_threshold: Optional[float] = None
    max_provisional_duration_sec: float = 2.5
    fixed_delay_updates: int = 3
