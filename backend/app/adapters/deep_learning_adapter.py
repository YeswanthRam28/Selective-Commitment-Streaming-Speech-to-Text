import os
import asyncio
import time
import numpy as np
import torch
from typing import AsyncGenerator, Optional, List
from app.adapters.base import BaseRecognizerAdapter
from app.models.events import HypothesisEvent, WordItem
from app.ml.model import StreamingCTCAcousticModel, extract_mel_spectrogram, CTCGreedyDecoder
from app.audio_processing import clean_speech_audio
from app.ml.language_model import CTCBeamSearchDecoder

_SHARED_WAV2VEC2_PROCESSOR = None
_SHARED_WAV2VEC2_MODEL = None

class DeepLearningRecognizer(BaseRecognizerAdapter):
    """
    On-device Deep Learning Streaming Speech Recognizer.
    Executes an acoustic neural network on the local NVIDIA RTX 3050 GPU (CUDA)
    without third-party cloud APIs.
    """
    @classmethod
    def preload(cls, checkpoint_path: Optional[str] = None):
        """Pre-loads and warms up Wav2Vec2 on CUDA GPU at server startup."""
        global _SHARED_WAV2VEC2_PROCESSOR, _SHARED_WAV2VEC2_MODEL
        if _SHARED_WAV2VEC2_MODEL is not None and _SHARED_WAV2VEC2_PROCESSOR is not None:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"[Startup Preload] Loading Wav2Vec2 CTC Model onto {device.type.upper()} ({device_name})...")
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            model_id = "facebook/wav2vec2-base-960h"
            _SHARED_WAV2VEC2_PROCESSOR = Wav2Vec2Processor.from_pretrained(model_id)
            _SHARED_WAV2VEC2_MODEL = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
            _SHARED_WAV2VEC2_MODEL.eval()

            # Warm-up pass to compile CUDA kernels and initialize memory allocations
            dummy_pcm = np.zeros(16000, dtype=np.float32)
            inputs = _SHARED_WAV2VEC2_PROCESSOR(dummy_pcm, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                _SHARED_WAV2VEC2_MODEL(inputs.input_values.to(device))
            print(f"[Startup Preload] Wav2Vec2 CTC Model pre-warmed & ready in GPU memory!")
        except Exception as e:
            print(f"[Startup Preload Warning] Failed to preload Wav2Vec2: {e}")
            raise e

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
        self.use_hf_pretrained = True

        self._init_model(checkpoint_path)

    def _init_model(self, checkpoint_path: Optional[str]):
        """Initializes PyTorch deep learning weights using preloaded cache when possible."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_ckpt = os.path.join(base_dir, "ml", "checkpoints", "model.pt")
        custom_ckpt = checkpoint_path or default_ckpt
        use_custom = (checkpoint_path is not None) or (os.getenv("USE_CUSTOM_MODEL") == "1")

        if use_custom and os.path.exists(custom_ckpt):
            try:
                print(f"📦 Loading custom Deep Learning CTC checkpoint from {custom_ckpt} onto {self.device}...")
                self.model = StreamingCTCAcousticModel().to(self.device)
                self.model.load_state_dict(torch.load(custom_ckpt, map_location=self.device))
                self.model.eval()
                self.use_hf_pretrained = False
                print("✅ Custom Acoustic Model checkpoint loaded successfully!")
                return
            except Exception as e:
                print(f"Failed to load custom checkpoint, falling back to pre-trained model: {e}")

        # Reuse preloaded Wav2Vec2
        global _SHARED_WAV2VEC2_PROCESSOR, _SHARED_WAV2VEC2_MODEL
        if _SHARED_WAV2VEC2_MODEL is None or _SHARED_WAV2VEC2_PROCESSOR is None:
            self.preload()

        self.hf_processor = _SHARED_WAV2VEC2_PROCESSOR
        self.hf_model = _SHARED_WAV2VEC2_MODEL
        self.use_hf_pretrained = True

        # Initialize CTC Beam Search Language Model Decoder
        vocab = self.hf_processor.tokenizer.get_vocab()
        id_to_char = {v: k for k, v in vocab.items()}
        pad_id = getattr(self.hf_processor.tokenizer, "pad_token_id", 0)
        self.beam_decoder = CTCBeamSearchDecoder(
            id_to_char=id_to_char,
            blank_id=pad_id,
            space_char="|",
            beam_width=16,
            lm_weight=0.9,
            word_bonus=1.2
        )

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
            await asyncio.sleep(0.35) # ~350ms streaming cadence

            if len(self.audio_bytes_buffer) < 4800: # at least 150ms of 16kHz audio
                continue

            # Convert 16-bit PCM bytes to float32 NumPy array
            audio_np = np.frombuffer(bytes(self.audio_bytes_buffer), dtype=np.int16).astype(np.float32) / 32768.0

            # Run GPU Deep Learning inference
            text, words = self._infer(audio_np)

            if text and text != last_decoded_text:
                print(f"🗣️ [GPU Decoded]: \"{text}\"")
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

    def _trim_leading_silence(self, audio_np: np.ndarray, threshold: float = 0.008) -> np.ndarray:
        """Trims leading silence to prevent CTC hallucination on quiet room noise."""
        if len(audio_np) == 0:
            return audio_np
        frame_len = 320 # 20ms at 16kHz
        num_frames = len(audio_np) // frame_len
        for f in range(num_frames):
            frame = audio_np[f * frame_len : (f + 1) * frame_len]
            if np.sqrt(np.mean(frame**2)) > threshold:
                start_idx = max(0, f * frame_len - 1600) # 100ms acoustic cushion
                return audio_np[start_idx:]
        return audio_np

    def _infer(self, audio_np: np.ndarray) -> tuple[str, List[dict]]:
        """Executes forward pass of deep learning network on GPU with audio denoising."""
        # Silence gating: ignore pure background noise/hiss
        rms = np.sqrt(np.mean(audio_np**2))
        if rms < 0.005:
            return "", []

        # Remove laptop fan noise and out-of-band hiss
        audio_np = clean_speech_audio(audio_np, sr=16000)

        # Trim leading quiet frames to avoid hallucinating consonants on room hiss
        audio_np = self._trim_leading_silence(audio_np)
        if len(audio_np) < 1600: # at least 100ms
            return "", []

        with torch.no_grad():
            if self.use_hf_pretrained and self.hf_model is not None:
                inputs = self.hf_processor(audio_np, sampling_rate=16000, return_tensors="pt")
                input_values = inputs.input_values.to(self.device)

                logits = self.hf_model(input_values).logits # (1, Time, Vocab)
                log_probs = torch.log_softmax(logits[0], dim=-1) # (Time, Vocab)

                # Beam search decoding with N-gram Language Model
                text, words = self.beam_decoder.decode(log_probs)
                if not text:
                    return "", []

                return text.lower().strip(), words
            else:
                # Custom StreamingCTCAcousticModel forward pass
                audio_tensor = torch.tensor(audio_np, dtype=torch.float32, device=self.device)
                mel = extract_mel_spectrogram(audio_tensor) # (1, 80, Time)
                log_probs = self.model(mel) # (Time_sub, 1, Vocab)
                text, words = CTCGreedyDecoder.decode(log_probs)
                return text.lower().strip(), words

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
        if len(self.audio_bytes_buffer) >= 3200:
            audio_np = np.frombuffer(bytes(self.audio_bytes_buffer), dtype=np.int16).astype(np.float32) / 32768.0
            final_text, words = self._infer(audio_np)
            if final_text:
                print(f"🏁 [GPU Final Utterance]: \"{final_text}\"")
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