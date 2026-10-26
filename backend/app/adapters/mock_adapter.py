import asyncio
import time
from typing import AsyncGenerator, List
from app.adapters.base import BaseRecognizerAdapter
from app.models.events import HypothesisEvent, WordItem

# Deterministic test script with intentional partial revisions
DEFAULT_MOCK_SCRIPT = [
    ("We", False, [0.95]),
    ("We are", False, [0.98, 0.96]),
    ("We are test", False, [0.99, 0.98, 0.60]),
    ("We are testing", False, [0.99, 0.99, 0.94]), # 'test' revised to 'testing'
    ("We are testing selective", False, [0.99, 0.99, 0.96, 0.91]),
    ("We are testing selective commitment", False, [0.99, 0.99, 0.97, 0.95, 0.93]),
    ("We are testing selective commitment in", False, [0.99, 0.99, 0.98, 0.96, 0.94, 0.90]),
    ("We are testing selective commitment in speech", False, [0.99, 0.99, 0.98, 0.96, 0.95, 0.92, 0.88]),
    ("We are testing selective commitment in speech recognition.", True, [0.99, 0.99, 0.98, 0.97, 0.96, 0.95, 0.96, 0.98])
]

class DeterministicMockRecognizer(BaseRecognizerAdapter):
    def __init__(self, script: List[tuple] = None, interval: float = 0.5):
        self.script = script or DEFAULT_MOCK_SCRIPT
        self.interval = interval
        self.session_id: str = ""
        self.seq: int = 0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self._playback_task: asyncio.Task = None

    async def start_session(self, session_id: str, sample_rate: int = 16000) -> None:
        self.session_id = session_id
        self.seq = 0
        self.is_running = True
        self.queue = asyncio.Queue()
        self._playback_task = asyncio.create_task(self._run_playback())

    async def _run_playback(self):
        for raw_text, is_final, confidences in self.script:
            if not self.is_running:
                break
            await asyncio.sleep(self.interval)
            words = raw_text.split()
            word_items = []
            for idx, w in enumerate(words):
                conf = confidences[idx] if idx < len(confidences) else 0.90
                word_items.append(WordItem(word=w, confidence=conf))

            self.seq += 1
            event = HypothesisEvent(
                session_id=self.session_id,
                seq=self.seq,
                is_final=is_final,
                words=word_items,
                raw_text=raw_text,
                timestamp=time.time()
            )
            await self.queue.put(event)
            if is_final:
                break

    async def send_audio(self, chunk_bytes: bytes) -> None:
        # In mock mode, audio ingestion keeps session active
        pass

    async def receive_events(self) -> AsyncGenerator[HypothesisEvent, None]:
        while self.is_running or not self.queue.empty():
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                yield event
                if event.is_final:
                    break
            except asyncio.TimeoutError:
                if not self.is_running:
                    break

    async def end_session(self) -> None:
        self.is_running = False
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
