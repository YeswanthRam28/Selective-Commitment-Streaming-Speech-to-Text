import numpy as np
import scipy.signal as signal

def bandpass_filter(audio_np: np.ndarray, sr: int = 16000, low: float = 85.0, high: float = 7500.0) -> np.ndarray:
    """
    4th-order Butterworth bandpass filter to eliminate low-frequency rumble (fan vibration, AC hum)
    and high-frequency mic hiss.
    """
    if len(audio_np) < 64:
        return audio_np
    try:
        sos = signal.butter(4, [low, high], btype='bandpass', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, audio_np)
        return filtered.astype(np.float32)
    except Exception:
        return audio_np

def spectral_noise_subtraction(audio_np: np.ndarray, sr: int = 16000, n_fft: int = 512, hop_length: int = 256) -> np.ndarray:
    """
    Suppresses stationary background noise (laptop GPU fan whine, ambient room hum)
    using spectral subtraction gating.
    """
    if len(audio_np) < n_fft * 2:
        return audio_np

    try:
        _, _, Zxx = signal.stft(audio_np, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate stationary noise spectrum from the lowest 15% energy frames
        frame_energies = np.sum(mag**2, axis=0)
        k = max(2, int(len(frame_energies) * 0.15))
        noise_idx = np.argsort(frame_energies)[:k]
        noise_spectrum = np.mean(mag[:, noise_idx], axis=1, keepdims=True)

        # Oversubtraction with spectral floor
        alpha = 1.6 # subtraction factor
        beta = 0.05  # minimum spectral floor to prevent musical noise artifacts
        clean_mag = np.maximum(mag - alpha * noise_spectrum, beta * mag)

        # Invert back to time domain
        clean_Zxx = clean_mag * np.exp(1j * phase)
        _, clean_audio = signal.istft(clean_Zxx, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        return clean_audio[:len(audio_np)].astype(np.float32)
    except Exception:
        return audio_np

def clean_speech_audio(audio_np: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Speech audio pre-processing pipeline for neural acoustic models:
    1. 4th-order Butterworth bandpass (85 Hz - 7500 Hz) eliminates mic rumble and Nyquist artifacts.
    2. Preserves natural unvoiced consonants (stops/fricatives like 't', 's', 'v') without spectral distortion.
    """
    if len(audio_np) == 0:
        return audio_np

    # Apply bandpass filter - transparent to human voice, removes room vibration & AC hum
    return bandpass_filter(audio_np, sr=sr)
