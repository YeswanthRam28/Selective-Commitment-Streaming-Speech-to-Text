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
    Full audio denoising pipeline:
    1. Bandpass filter (85 Hz - 7500 Hz)
    2. Spectral stationary noise reduction (fan/AC whirring suppression)
    """
    if len(audio_np) == 0:
        return audio_np

    # Step 1: Remove out-of-band noise
    bp = bandpass_filter(audio_np, sr=sr)

    # Step 2: Subtract stationary fan noise
    clean = spectral_noise_subtraction(bp, sr=sr)

    return clean
