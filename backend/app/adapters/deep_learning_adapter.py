import os
import asyncio
import time
import numpy as np
import torch
from typing import AsyncGenerator, Optional, List
from app.adapters.base import BaseRecognizerAdapter
from app.models.events import HypothesisEvent, WordItem
from app.ml.model import StreamingCTCAcousticModel, extract_mel_spectrogram, CTCGreedyDecoder

class DeepLearningRecognizer(BaseRecognizerAdapter):
    """
    On-device Deep Learning Streaming Speech Recognizer.
    Executes an acoustic neural network on the local NVIDIA RTX 3050 GPU (CUDA)
    without third-party cloud APIs.
    """
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.session_id: str = ""
        self.seq: int = 0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.audio_bytes_buffer: bytearray = bytearray()
        self.is_running: bool = False
        self._inference_task: Optional[asyncio.Task] = None
        self.model = None
        self.hf_processor = None
        self.hf_model = None
        self.use_hf_pretrained = False

        self._init_model(checkpoint_path)

    def _init_model(self, checkpoint_path: Optional[str]):
        """
        Initializes PyTorch deep learning weights:
        1. If a local trained checkpoint exists, loads StreamingCTCAcousticModel.
        2. Else loads standard Wav2Vec2-CTC PyTorch weights onto CUDA for instant high-accuracy mic recognition.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_ckpt = os.path.join(base_dir, "ml", "checkpoints", "model.pt")
        custom_ckpt = checkpoint_path or default_ckpt

        if torch.cuda.is_available():
            print(f"🚀 Initializing Deep Learning Model on CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA not available in this Python environment. Running on CPU.")

        if os.path.exists(custom_ckpt):
            try:
                print(f"📦 Loading custom Deep Learning CTC checkpoint from {custom_ckpt} onto {self.device}...")
                self.model = StreamingCTCAcousticModel().to(self.device)
                self.model.load_state_dict(torch.load(custom_ckpt, map_location=self.device))
                self.model.eval()
                self.use_hf_pretrained = False
                return
            except Exception as e:
                print(f"Failed to load custom checkpoint: {e}")

        # Load pure PyTorch CTC Model (Wav2Vec2 architecture)
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            print(f" Loading Deep Learning CTC Acoustic Model onto {self.device}...")
            model_id = "facebook/wav2vec2-base-960h"
            self.hf_processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.hf_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
            self.hf_model.eval()
            self.use_hf_pretrained = True
            print(f" Deep Learning model loaded on {self.device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        except Exception as e:
            print(f" Falling back to custom architecture: {e}")
            self.model = StreamingCTCAcousticModel().to(self.device)
            self.model.eval()
            self.use_hf_pretrained = False

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

    async def _streaming_inference_loop(self):
        """
        Periodically runs the deep learning acoustic network over the buffered audio.
        Emits incremental partial hypotheses to the Selective Commitment policy.
        """
        last_decoded_text = ""
        while self.is_running:
            await asyncio.sleep(0.30) # ~300ms streaming cadence

            if len(self.audio_bytes_buffer) < 3200: # at least 100ms of 16kHz audio
                continue

            # Convert 16-bit PCM bytes to float32 NumPy array
            audio_np = np.frombuffer(bytes(self.audio_bytes_buffer), dtype=np.int16).astype(np.float32) / 32768.0

            # Run GPU Deep Learning inference
            text, words = self._infer(audio_np)

            if text and text != last_decoded_text:
                last_decoded_text = text
                self.seq += 1
                word_items = [
                    WordItem(
                        word=w["word"],
                        start_time=w.get("start_time"),
                        end_time=w.get("end_time"),
                        confidence=w.get("confidence", 0.90)
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

    def _infer(self, audio_np: np.ndarray) -> tuple[str, List[dict]]:
        """Executes forward pass of deep learning network on GPU."""
        with torch.no_grad():
            if self.use_hf_pretrained and self.hf_model is not None:
                inputs = self.hf_processor(audio_np, sampling_rate=16000, return_tensors="pt")
                input_values = inputs.input_values.to(self.device)

                logits = self.hf_model(input_values).logits # (1, Time, Vocab)
                pred_ids = torch.argmax(logits, dim=-1)[0]
                text = self.hf_processor.decode(pred_ids)

                probs = torch.softmax(logits, dim=-1)
                confs = torch.max(probs, dim=-1).values[0].cpu().tolist()

                words = []
                for w in text.split():
                    words.append({
                        "word": w.strip().lower(),
                        "confidence": 0.92
                    })
                return text.lower(), words
            else:
                # Custom StreamingCTCAcousticModel forward pass
                audio_tensor = torch.tensor(audio_np, dtype=torch.float32, device=self.device)
                mel = extract_mel_spectrogram(audio_tensor) # (1, 80, Time)
                log_probs = self.model(mel) # (Time_sub, 1, Vocab)
                text, words = CTCGreedyDecoder.decode(log_probs)
                return text, words

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

        # Emit final reconciled hypothesis
        if len(self.audio_bytes_buffer) >= 1600:
            audio_np = np.frombuffer(bytes(self.audio_bytes_buffer), dtype=np.int16).astype(np.float32) / 32768.0
            final_text, words = self._infer(audio_np)
            if final_text:
                self.seq += 1
                word_items = [
                    WordItem(word=w["word"], confidence=w.get("confidence", 0.95))
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
