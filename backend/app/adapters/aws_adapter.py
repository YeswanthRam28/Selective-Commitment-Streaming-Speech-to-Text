import os
import asyncio
import time
from typing import AsyncGenerator, Optional
from app.adapters.base import BaseRecognizerAdapter
from app.models.events import HypothesisEvent, WordItem

try:
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

class AWSTranscribeAdapter(BaseRecognizerAdapter):
    def __init__(self, region: Optional[str] = None):
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.session_id: str = ""
        self.seq: int = 0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self._stream = None
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None

    async def start_session(self, session_id: str, sample_rate: int = 16000) -> None:
        if not AWS_AVAILABLE:
            raise RuntimeError("amazon-transcribe package is not installed.")
        
        self.session_id = session_id
        self.seq = 0
        self.is_running = True
        self.queue = asyncio.Queue()
        self._audio_queue = asyncio.Queue()

        client = TranscribeStreamingClient(region=self.region)
        self._stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=sample_rate,
            media_encoding="pcm"
        )

        class InternalHandler(TranscriptResultStreamHandler):
            def __init__(handler_self, outer_self):
                super().__init__(outer_self._stream.output_stream)
                handler_self.outer = outer_self

            async def handle_transcript_event(handler_self, transcript_event: TranscriptEvent):
                results = transcript_event.transcript.results
                for res in results:
                    is_final = not res.is_partial
                    for alt in res.alternatives:
                        raw_text = alt.transcript
                        words = []
                        if alt.items:
                            for item in alt.items:
                                if item.content:
                                    conf = getattr(item, 'confidence', None)
                                    words.append(WordItem(
                                        word=item.content,
                                        start_time=getattr(item, 'start_time', None),
                                        end_time=getattr(item, 'end_time', None),
                                        confidence=conf
                                    ))
                        else:
                            words = [WordItem(word=w) for w in raw_text.split()]

                        handler_self.outer.seq += 1
                        event = HypothesisEvent(
                            session_id=handler_self.outer.session_id,
                            seq=handler_self.outer.seq,
                            is_final=is_final,
                            words=words,
                            raw_text=raw_text,
                            timestamp=time.time()
                        )
                        await handler_self.outer.queue.put(event)

        handler = InternalHandler(self)

        async def _audio_sender():
            try:
                while self.is_running:
                    chunk = await self._audio_queue.get()
                    if chunk is None:
                        break
                    await self._stream.input_stream.send_audio_event(audio_chunk=chunk)
                await self._stream.input_stream.end_stream()
            except Exception as e:
                pass

        async def _event_receiver():
            try:
                await handler.handle_events()
            except Exception as e:
                pass

        self._sender_task = asyncio.create_task(_audio_sender())
        self._receiver_task = asyncio.create_task(_event_receiver())

    async def send_audio(self, chunk_bytes: bytes) -> None:
        if self.is_running:
            await self._audio_queue.put(chunk_bytes)

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
        await self._audio_queue.put(None)
        if self._sender_task:
            await self._sender_task
        if self._receiver_task:
            await self._receiver_task
