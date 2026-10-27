import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import List, Tuple, Optional

from app.ml.model import StreamingCTCAcousticModel, extract_mel_spectrogram
from app.ml.vocabulary import text_to_ids, VOCAB_SIZE, BLANK_ID, ids_to_text
from app.ml.dataset import RealSpeechDataset, SpeechDataAugmentation
from app.metrics.evaluator import compute_wer

def pad_custom_collate_fn(batch):
    """
    Collates a batch of (waveform, text) pairs for the custom StreamingCTCAcousticModel.
    Converts waveforms to Mel-spectrograms and pads for CTC Loss.
    """
    mels = []
    targets = []
    for waveform, text in batch:
        mel = extract_mel_spectrogram(waveform.unsqueeze(0)).squeeze(0) # (80, Time)
        mels.append(mel)
        targets.append(torch.tensor(text_to_ids(text), dtype=torch.long))

    mel_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    max_mel_len = max(mel_lengths).item()
    padded_mels = torch.zeros(len(mels), 80, max_mel_len)
    for i, m in enumerate(mels):
        padded_mels[i, :, :m.shape[1]] = m

    flat_targets = torch.cat(targets)
    return padded_mels, flat_targets, mel_lengths, target_lengths

def train_custom_model(
    dataset: RealSpeechDataset,
    num_epochs: int = 10,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "app/ml/checkpoints/custom_model.pt"
):
    """
    Trains the StreamingCTCAcousticModel from scratch on real speech with SpecAugment and CTCLoss.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    augmenter = SpeechDataAugmentation()

    # Split train / validation (90% / 10%)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_custom_collate_fn)

    model = StreamingCTCAcousticModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    print(f"\n[Training] Starting Custom CTC Acoustic Model Training on {device.upper()}...")
    print(f"[Training] Dataset size: {len(dataset)} (Train: {len(train_ds)}, Val: {len(val_ds)}) | Epochs: {num_epochs}")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        start_t = time.time()

        for batch_idx, (mels, flat_targets, mel_lengths, target_lengths) in enumerate(train_loader):
            # Apply SpecAugment during training
            mels = augmenter(mels, p=0.6).to(device)
            flat_targets = flat_targets.to(device)

            optimizer.zero_grad()
            log_probs = model(mels) # (Time_sub, Batch, Vocab)
            input_lengths = torch.clamp(mel_lengths // 2, min=1).to(device)

            loss = criterion(log_probs, flat_targets, input_lengths, target_lengths.to(device))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        # Validation pass
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for mels, flat_targets, mel_lengths, target_lengths in val_loader:
                mels = mels.to(device)
                flat_targets = flat_targets.to(device)
                log_probs = model(mels)
                input_lengths = torch.clamp(mel_lengths // 2, min=1).to(device)
                val_loss = criterion(log_probs, flat_targets, input_lengths, target_lengths.to(device))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        elapsed = time.time() - start_t

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Checkpoint saved to: {save_path} (Val Loss: {best_val_loss:.4f})")

    print(f"[Training Complete] Custom Model saved successfully to {save_path}!\n")

def finetune_wav2vec2(
    dataset: RealSpeechDataset,
    model_id: str = "facebook/wav2vec2-base-960h",
    num_epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "app/ml/checkpoints/finetuned_wav2vec2.pt"
):
    """
    Fine-tunes the Wav2Vec2 CTC Acoustic Model on real speech audio on CUDA GPU.
    Freezes 1D CNN feature extractor to preserve base acoustic representations,
    and updates contextual Transformer layers for high domain accuracy.
    """
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n[Fine-Tuning] Loading {model_id} for GPU transfer learning...")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

    # Freeze low-level convolutional feature extractor to maintain robust acoustic foundation
    model.freeze_feature_encoder()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    def wav2vec_collate(batch):
        waveforms, texts = zip(*batch)
        inputs = processor([w.numpy() for w in waveforms], sampling_rate=16000, padding=True, return_tensors="pt")
        with processor.as_target_processor():
            labels = processor(list(texts), padding=True, return_tensors="pt").input_ids
        # Replace padding with -100 to ignore in CTC loss
        labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
        return inputs.input_values, labels

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=wav2vec_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=wav2vec_collate)

    print(f"[Fine-Tuning] Starting transfer learning on {device.upper()}...")
    print(f"[Fine-Tuning] Samples: {len(dataset)} | Batch Size: {batch_size} | Learning Rate: {lr}")

    best_wer = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for batch_idx, (input_values, labels) in enumerate(train_loader):
            input_values = input_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        # Validation evaluation & WER computation
        model.eval()
        total_wer = 0.0
        val_samples = 0
        with torch.no_grad():
            for input_values, labels in val_loader:
                input_values = input_values.to(device)
                logits = model(input_values).logits
                pred_ids = torch.argmax(logits, dim=-1)
                pred_texts = processor.batch_decode(pred_ids)

                # Decode labels
                labels[labels == -100] = processor.tokenizer.pad_token_id
                target_texts = processor.batch_decode(labels, group_tokens=False)

                for ref, hyp in zip(target_texts, pred_texts):
                    wer_metrics = compute_wer(ref, hyp)
                    total_wer += wer_metrics["wer"]
                    val_samples += 1

        avg_wer = (total_wer / max(val_samples, 1)) * 100
        elapsed = time.time() - start_t

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {avg_loss:.4f} | Val WER: {avg_wer:.2f}% | Time: {elapsed:.2f}s")

        if avg_wer < best_wer:
            best_wer = avg_wer
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved to: {save_path} (Val WER: {best_wer:.2f}%)")

    print(f"[Fine-Tuning Complete] Model saved successfully to {save_path}!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selective Commitment Speech ASR Training")
    parser.add_argument("--mode", type=str, choices=["custom", "finetune"], default="finetune",
                        help="Train custom BiLSTM or fine-tune Wav2Vec2 on real speech")
    parser.add_argument("--dataset", type=str, default="dev-clean",
                        help="LibriSpeech subset ('dev-clean', 'test-clean') or 'custom'")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Load real speech dataset
    speech_ds = RealSpeechDataset(dataset_type="librispeech", url=args.dataset, download=True)

    if args.mode == "finetune":
        finetune_wav2vec2(speech_ds, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        train_custom_model(speech_ds, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
