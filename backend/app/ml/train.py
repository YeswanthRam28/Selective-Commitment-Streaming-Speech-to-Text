import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from app.ml.model import StreamingCTCAcousticModel, extract_mel_spectrogram
from app.ml.vocabulary import text_to_ids, VOCAB_SIZE, BLANK_ID

class SpeechSampleDataset(Dataset):
    """
    Dataset loader for speech-to-text pairs.
    Can ingest audio waveforms and string references.
    """
    def __init__(self, samples: List[Tuple[torch.Tensor, str]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, text = self.samples[idx]
        # Waveform shape: (1, Num_Samples)
        mel = extract_mel_spectrogram(waveform) # (1, 80, Time)
        target_ids = torch.tensor(text_to_ids(text), dtype=torch.long)
        return mel.squeeze(0), target_ids

def pad_collate_fn(batch):
    """
    Collates a batch of (mel_spec, target_ids) with padding for CTC training.
    """
    mels, targets = zip(*batch)
    mel_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    max_mel_len = max(mel_lengths).item()
    padded_mels = torch.zeros(len(mels), 80, max_mel_len)
    for i, m in enumerate(mels):
        padded_mels[i, :, :m.shape[1]] = m

    flat_targets = torch.cat(targets)
    return padded_mels, flat_targets, mel_lengths, target_lengths

def train_ctc_model(
    model: StreamingCTCAcousticModel,
    dataloader: DataLoader,
    num_epochs: int = 15,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "backend/app/ml/checkpoints/model.pt"
):
    """
    Trains the StreamingCTCAcousticModel using CTC loss on the given GPU/CPU device.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    print(f" Starting CTC Acoustic Model Training on device: {device.upper()}")
    model.train()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        start_t = time.time()

        for batch_idx, (mels, flat_targets, mel_lengths, target_lengths) in enumerate(dataloader):
            mels = mels.to(device)
            flat_targets = flat_targets.to(device)

            optimizer.zero_grad()
            # Forward pass: (Time_sub, Batch, Vocab)
            log_probs = model(mels)

            # Conv layer downsamples time by 2x
            input_lengths = torch.clamp(mel_lengths // 2, min=1).to(device)

            loss = criterion(log_probs, flat_targets, input_lengths, target_lengths.to(device))
            loss.backward()

            # Gradient clipping for stable recurrent optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        elapsed = time.time() - start_t
        print(f"Epoch {epoch:02d}/{num_epochs:02d} | CTC Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

    torch.save(model.state_dict(), save_path)
    print(f" Training complete. Model checkpoint saved to: {save_path}")
    return model

if __name__ == "__main__":
    # Self-test synthetic speech dataset generator
    torch.manual_seed(42)
    synthetic_samples = []
    test_phrases = [
        "we are testing speech recognition",
        "selective commitment policy in real time",
        "deep learning model running on gpu",
        "audio streaming pipeline with ctc loss"
    ]
    for phrase in test_phrases:
        # 2 seconds of synthetic audio @ 16kHz
        dummy_audio = torch.randn(1, 32000)
        synthetic_samples.append((dummy_audio, phrase))

    ds = SpeechSampleDataset(synthetic_samples)
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=pad_collate_fn)

    net = StreamingCTCAcousticModel()
    train_ctc_model(net, dl, num_epochs=5)
