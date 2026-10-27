import os
import subprocess
import soundfile as sf
import torchaudio
import torch
import numpy as np

def generate_speech_dataset():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "custom_speech")
    os.makedirs(data_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, "manifest.csv")

    phrases = [
        "the live incremental transcript",
        "live incremental transcript",
        "hello this is a test evaluation",
        "this is a test evaluation",
        "can you hear me",
        "selective commitment streaming speech to text",
        "selective commitment",
        "streaming speech recognition",
        "testing speech recognition on deep learning model",
        "low latency streaming speech to text",
        "the quick brown fox jumps over the lazy dog",
        "zero latency streaming speech recognition",
        "acoustic model evaluation on gpu",
        "we are testing the live transcription system",
        "provisional and committed transcript updates"
    ]

    # Rates: -3 to +3
    rates = [-2, -1, 0, 1, 2]
    
    ps_script_path = os.path.join(data_dir, "synth.ps1")
    manifest_lines = []
    sample_idx = 0

    print(f"Generating synthetic speech dataset with acoustic variations in {data_dir}...")

    # Write powershell synthesis script
    with open(ps_script_path, "w", encoding="utf-8") as f:
        f.write("""
param([string]$Text, [string]$OutPath, [int]$Rate)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = $Rate
$synth.SetOutputToWaveFile($OutPath)
$synth.Speak($Text)
$synth.Dispose()
""")

    for phrase in phrases:
        for rate in rates:
            raw_wav = os.path.join(data_dir, f"sample_{sample_idx:04d}_raw.wav")
            final_wav = os.path.join(data_dir, f"sample_{sample_idx:04d}.wav")

            # Run synthesis
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_script_path, "-Text", phrase, "-OutPath", raw_wav, "-Rate", str(rate)]
            res = subprocess.run(cmd, capture_output=True, text=True)

            if os.path.exists(raw_wav):
                # Load and resample to 16kHz mono, add light acoustic variations
                waveform, sr = torchaudio.load(raw_wav)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Save 16kHz mono wav
                torchaudio.save(final_wav, waveform, 16000)
                try:
                    os.remove(raw_wav)
                except Exception:
                    pass

                manifest_lines.append(f"{final_wav},{phrase}\n")
                sample_idx += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.writelines(manifest_lines)

    print(f"Generated {len(manifest_lines)} real audio samples in {manifest_path}!")

if __name__ == "__main__":
    generate_speech_dataset()
