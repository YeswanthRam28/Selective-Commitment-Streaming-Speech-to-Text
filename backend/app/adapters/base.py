from abc import ABC, abstractmethod
from typing import AsyncGenerator
from app.models.events import HypothesisEvent

class BaseRecognizerAdapter(ABC):
    """
    Provider-neutral interface for streaming speech recognizers.
    """
    @abstractmethod
    async def start_session(self, session_id: str, sample_rate: int = 16000) -> None:
        """Initialize the recognizer session."""
        pass

    @abstractmethod
    async def send_audio(self, chunk_bytes: bytes) -> None:
        """Send an audio chunk (16-bit PCM mono)."""
        pass

    @abstractmethod
    async def receive_events(self) -> AsyncGenerator[HypothesisEvent, None]:
        """Asynchronously yield incoming partial and final hypotheses."""
        pass

    @abstractmethod
    async def end_session(self) -> None:
        """Finalize recognition and flush remaining hypotheses."""
        pass
