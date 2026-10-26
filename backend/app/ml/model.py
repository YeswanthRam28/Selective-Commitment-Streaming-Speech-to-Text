import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from app.ml.vocabulary import VOCAB_SIZE, BLANK_ID, ID_TO_CHAR, SPACE_ID

class ConvFeatureExtractor(nn.Module):
    """
    Subsampling 1D Convolutional block for acoustic feature extraction.
    Compresses temporal resolution by factor of 2x or 4x.
    """
    def __init__(self, in_channels: int = 80, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Channels/Mels, Time)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        # Output shape: (Batch, Hidden, Time/2)
        return x

class StreamingCTCAcousticModel(nn.Module):
    """
    Deep Neural Network Acoustic Model for Streaming Speech Recognition.
    Architecture:
      1. Convolutional Front-End (subsamples Mel frames)
      2. Multi-layer Bidirectional LSTM Encoder
      3. Linear Projection Head -> CTC Vocabulary Logits
    """
    def __init__(
        self,
        n_mels: int = 80,
        conv_dim: int = 256,
        rnn_hidden: int = 256,
        rnn_layers: int = 3,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_mels = n_mels
        self.conv = ConvFeatureExtractor(in_channels=n_mels, hidden_dim=conv_dim)
        
        self.lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )
        
        self.layer_norm = nn.LayerNorm(rnn_hidden * 2)
        self.fc = nn.Linear(rnn_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Mel spectrogram tensor of shape (Batch, n_mels, Time)
        Returns:
            Log-probabilities of shape (Time, Batch, vocab_size) suitable for nn.CTCLoss
        """
        # 1. Convolutional subsampling
        features = self.conv(x) # (Batch, conv_dim, Time_sub)
        
        # 2. Transpose to (Batch, Time_sub, conv_dim) for LSTM
        features = features.transpose(1, 2)
        
        # 3. Recurrent encoder
        lstm_out, _ = self.lstm(features) # (Batch, Time_sub, rnn_hidden*2)
        lstm_out = self.layer_norm(lstm_out)
        
        # 4. CTC projection head
        logits = self.fc(lstm_out) # (Batch, Time_sub, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Return in (Time, Batch, vocab_size) format for PyTorch CTCLoss
        return log_probs.transpose(0, 1)

def extract_mel_spectrogram(
    audio_tensor: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160
) -> torch.Tensor:
    """
    Extracts Log-Mel Spectrogram from raw 16kHz audio tensor using pure PyTorch.
    Audio: shape (Batch, Num_Samples) or (Num_Samples,)
    """
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    window = torch.hann_window(n_fft, device=audio_tensor.device)
    # STFT
    stft = torch.stft(
        audio_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True
    )
    magnitudes = stft.abs() ** 2

    # Construct Mel filterbank matrix in PyTorch
    mel_fb = torch.tensor(
        _create_mel_filterbank(sample_rate, n_fft, n_mels),
        dtype=torch.float32,
        device=audio_tensor.device
    )

    # (Batch, n_mels, Time)
    mel_spec = torch.matmul(mel_fb, magnitudes)
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
    return log_mel

def _create_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int):
    """Generates standard triangular Mel filterbanks."""
    import numpy as np
    low_freq = 0.0
    high_freq = sample_rate / 2.0

    low_mel = 2595.0 * np.log10(1.0 + low_freq / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + high_freq / 700.0)

    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    freq_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * freq_points / sample_rate).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                fb[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                fb[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

    return fb

class CTCGreedyDecoder:
    """
    Greedy Best-Path Decoder for CTC outputs.
    Extracts decoded characters, collapses repeats, strips blank tokens,
    and returns word items with time and confidence estimates.
    """
    @staticmethod
    def decode(log_probs: torch.Tensor) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Args:
            log_probs: Tensor of shape (Time, 1, vocab_size) or (Time, vocab_size)
        Returns:
            (full_text, list_of_word_dicts)
        """
        if log_probs.dim() == 3:
            log_probs = log_probs.squeeze(1)

        probs = torch.exp(log_probs)
        best_ids = torch.argmax(log_probs, dim=-1).cpu().tolist()
        best_probs = torch.max(probs, dim=-1).values.cpu().tolist()

        collapsed_tokens = []
        collapsed_times = []
        collapsed_confs = []

        prev_id = -1
        for t, (tok_id, conf) in enumerate(zip(best_ids, best_probs)):
            if tok_id != prev_id:
                if tok_id != BLANK_ID:
                    collapsed_tokens.append(tok_id)
                    collapsed_times.append(t)
                    collapsed_confs.append(conf)
            prev_id = tok_id

        # Group tokens into words separated by SPACE_ID
        words: List[Dict[str, Any]] = []
        current_chars = []
        current_confs = []
        start_t = 0.0

        for tok_id, t, conf in zip(collapsed_tokens, collapsed_times, collapsed_confs):
            if tok_id == SPACE_ID:
                if current_chars:
                    word_str = "".join(current_chars).strip()
                    if word_str:
                        avg_conf = sum(current_confs) / len(current_confs)
                        words.append({
                            "word": word_str,
                            "start_time": round(start_t * 0.02, 3),
                            "end_time": round(t * 0.02, 3),
                            "confidence": round(avg_conf, 3)
                        })
                    current_chars = []
                    current_confs = []
            else:
                char = ID_TO_CHAR.get(tok_id, "")
                if not current_chars:
                    start_t = t
                current_chars.append(char)
                current_confs.append(conf)

        if current_chars:
            word_str = "".join(current_chars).strip()
            if word_str:
                avg_conf = sum(current_confs) / len(current_confs)
                words.append({
                    "word": word_str,
                    "start_time": round(start_t * 0.02, 3),
                    "end_time": round(len(best_ids) * 0.02, 3),
                    "confidence": round(avg_conf, 3)
                })

        full_text = " ".join(w["word"] for w in words)
        return full_text, words