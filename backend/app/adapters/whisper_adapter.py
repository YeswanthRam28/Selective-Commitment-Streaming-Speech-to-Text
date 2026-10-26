import os
import asyncio
import time
import numpy as np
import torch
from typing import AsyncGenerator, Optional, List
from app.adapters.base import BaseRecognizerAdapter
from app.models.events import HypothesisEvent, WordItem
from app.audio_processing import clean_speech_audio

class WhisperRecognizer(BaseRecognizerAdapter):
    """
    On-device OpenAI Whisper Streaming Speech Recognizer.
    Executes an encoder-decoder Transformer on the local NVIDIA RTX 3050 GPU (CUDA)
    for high-precision real-time streaming speech recognition.
    """
    def __init__(self, model_id: str = "openai/whisper-tiny.en"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.session_id: str = ""
        self.seq: int = 0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.audio_bytes_buffer: bytearray = bytearray()
        self.is_running: bool = False
        self._inference_task: Optional[asyncio.Task] = None
        self.processor = None
        self.model = None

        self._init_model()

    def _init_model(self):
        """Initializes Whisper model onto CUDA GPU."""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            print(f"[Whisper] Loading {self.model_id} onto {self.device.type.upper()} ({device_name})...")
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            print(f"[Whisper] Loaded {self.model_id} on {self.device.type.upper()} successfully!")
        except Exception as e:
            print(f"[Whisper] Failed to load model: {e}")
            raise e

    async def start_session(self, session_id: str, sample_rate: int = 16000) -> None:
        self.session_id = session_id
        self.seq = 0
        self.audio_bytes_buffer = bytearray()
        self.is_running = True
        self.queue = asyncio.Queue()
        self._inference_task = asyncio.create_task(self._streaming_inference_loop())

    async def send_audio(self, chunk_bytes: bytes) -> None:
        if self.is_running and chunk_bytes:
            self.audio_bytes_buffer.extend(chunk_bytes)

    def _trim_leading_silence(self, audio_np: np.ndarray, threshold: float = 0.008) -> np.ndarray:
        """Trims leading silence to prevent phantom hallucination on initial room noise."""
        if len(audio_np) == 0:
            return audio_np
        frame_len = 320 # 20ms at 16kHz
        num_frames = len(audio_np) // frame_len
        for f in range(num_frames):
            frame = audio_np[f * frame_len : (f + 1) * frame_len]
            if np.sqrt(np.mean(frame**2)) > threshold:
                start_idx = max(0, f * frame_len - 1600) # 100ms cushion
                return audio_np[start_idx:]
        return audio_np

    def _infer(self, audio_np: np.ndarray) -> tuple[str, List[dict]]:
        """Runs forward Whisper pass on GPU with audio denoising and anti-hallucination masks."""
        rms = np.sqrt(np.mean(audio_np**2))
        if rms < 0.005:
            return "", []

        # Step 1: Remove stationary laptop fan whine and out-of-band noise
        audio_np = clean_speech_audio(audio_np, sr=16000)

        # Step 2: Trim leading room silence
        audio_np = self._trim_leading_silence(audio_np)
        if len(audio_np) < 3200: # at least 200ms
            return "", []

        try:
            with torch.no_grad():
                # Extract log-mel filterbanks with explicit attention_mask for variable length audio
                inputs = self.processor(audio_np, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                input_features = inputs.input_features.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

                # Generate autoregressive tokens deterministically (temperature=0.0) without sampling hallucinations
                predicted_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    temperature=0.0,
                    max_new_tokens=48,
                    no_repeat_ngram_size=3
                )
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

                if not transcription:
                    return "", []

                # Filter common Whisper silence hallucinations
                lower_trans = transcription.lower()
                hallucination_phrases = ["subtitles by", "thank you for watching", "open ai", "amara.org", "subscribe"]
                if any(phrase in lower_trans for phrase in hallucination_phrases):
                    return "", []

                # Remove trailing punctuation to normalize into tokens for selective commitment
                clean_punct = transcription.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace("\"", "")
                raw_words = clean_punct.split()

                words = []
                for w in raw_words:
                    word_str = w.strip().lower()
                    if word_str:
                        words.append({
                            "word": word_str,
                            "confidence": 0.96
                        })

                return " ".join(raw_words).lower(), words
        except Exception as e:
            print(f"[Whisper Inference Error]: {e}")
            return "", []

    async def _streaming_inference_loop(self):
        """Periodically runs Whisper inference over streaming buffer."""
        last_decoded_text = ""
        while self.is_running:
            await asyncio.sleep(0.40) # 400ms cadence

            if len(self.audio_bytes_buffer) < 4800: # at least 150ms
                continue

            audio_np = np.frombuffer(bytes(self.audio_bytes_buffer), dtype=np.int16).astype(np.float32) / 32768.0
            text, words = self._infer(audio_np)

            if text and text != last_decoded_text:
                print(f"[Whisper GPU Decoded]: \"{text}\"")
                last_decoded_text = text
                self.seq += 1
                word_items = [
                    WordItem(
                        word=w["word"],
                        start_time=w.get("start_time"),
                        end_time=w.get("end_time"),
                        confidence=w.get("confidence", 0.95)
                    )
                    for w in words
                ]

                event = HypothesisEvent(
                    session_id=self.session_id,
                    seq=self.seq,
                    is_final=False,
                    words=word_items,
                    raw_text=text,
                    timestamp=time.time()
                )
                await self.queue.put(event)

    async def receive_events(self) -> AsyncGenerator[HypothesisEvent, None]:
        while self.is_running or not self.queue.empty():
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=0.8)
                yield event
                if event.is_final:
                    break
            except asyncio.TimeoutError:
                if not self.is_running:
                    break

    async def end_session(self) -> None:
        self.is_running = False
        if self._inference_task and not self._inference_task.done():
            self._inference_task.cancel()

        if len(self.audio_bytes_buffer) >= 3200:
            audio_np = np.frombuffer(bytes(self.audio_bytes_buffer), dtype=np.int16).astype(np.float32) / 32768.0
            final_text, words = self._infer(audio_np)
            if final_text:
                print(f"[Whisper Final Utterance]: \"{final_text}\"")
                self.seq += 1
                word_items = [
                    WordItem(word=w["word"], confidence=w.get("confidence", 0.98))
                    for w in words
                ]
                final_event = HypothesisEvent(
                    session_id=self.session_id,
                    seq=self.seq,
                    is_final=True,
                    words=word_items,
                    raw_text=final_text,
                    timestamp=time.time()
                )
                await self.queue.put(final_event)
