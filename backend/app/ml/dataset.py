import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from typing import List, Tuple, Optional, Dict

class SpeechDataAugmentation(nn.Module):
    """
    Real-time acoustic data augmentation for speech recognition training:
    1. SpecAugment (Frequency and Time masking)
    2. Additive background noise injection (for fan/room noise robustness)
    """
    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 25, noise_snr_db: Optional[float] = 20.0):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.noise_snr_db = noise_snr_db

    def forward(self, mel_spec: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """Applies SpecAugment masking to log-mel spectrogram with probability p."""
        if self.training and random.random() < p:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)
        return mel_spec

    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mixes low-level Gaussian noise into raw waveform during training."""
        if not self.training or self.noise_snr_db is None:
            return waveform
        signal_power = waveform.norm(p=2)
        noise = torch.randn_like(waveform)
        noise_power = noise.norm(p=2)
        snr = 10.0 ** (self.noise_snr_db / 20.0)
        scale = signal_power / (snr * (noise_power + 1e-7))
        return waveform + scale * noise

class RealSpeechDataset(Dataset):
    """
    Real-world speech dataset for training and fine-tuning acoustic models.
    Supports:
    1. LibriSpeech official benchmarks ('dev-clean', 'test-clean', 'train-clean-100')
    2. Local custom folder of audio files with CSV manifest ('filepath,transcript')
    """
    def __init__(
        self,
        dataset_type: str = "librispeech",
        root_dir: str = "data/librispeech",
        url: str = "dev-clean",
        download: bool = True,
        csv_path: Optional[str] = None,
        sample_rate: int = 16000,
        max_duration_sec: float = 15.0
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_sec * sample_rate)
        self.items: List[Tuple[str, str]] = [] # list of (audio_path, transcript)
        self.dataset_type = dataset_type

        os.makedirs(root_dir, exist_ok=True)

        if dataset_type == "librispeech":
            print(f"[Dataset] Loading LibriSpeech subset '{url}' from {root_dir}...")
            # Use torchaudio LIBRISPEECH loader
            self.librispeech = torchaudio.datasets.LIBRISPEECH(
                root=root_dir,
                url=url,
                download=download
            )
            print(f"[Dataset] LibriSpeech '{url}' loaded: {len(self.librispeech)} real audio recordings ready.")
        elif dataset_type == "custom" and csv_path and os.path.exists(csv_path):
            print(f"[Dataset] Ingesting custom speech dataset from {csv_path}...")
            with open(csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        audio_p, transcript = parts[0], ",".join(parts[1:])
                        if os.path.exists(audio_p):
                            self.items.append((audio_p, transcript.strip().lower()))
            print(f"[Dataset] Loaded {len(self.items)} custom speech samples.")
        else:
            self.librispeech = None

    def __len__(self) -> int:
        if self.dataset_type == "librispeech" and self.librispeech is not None:
            return len(self.librispeech)
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        if self.dataset_type == "librispeech" and self.librispeech is not None:
            # LibriSpeech returns: (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
            waveform, sr, transcript, _, _, _ = self.librispeech[idx]
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            # Ensure mono (1, Samples)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            text = transcript.strip().lower()
            return waveform.squeeze(0), text
        else:
            audio_path, text = self.items[idx]
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            return waveform.squeeze(0), text
