#!/usr/bin/env python3
"""
Synthetic audio dataset — generates simple audio with text descriptions.
No downloads needed. Uses numpy + scipy for audio synthesis.
"""

import math
import random
import torch
import numpy as np
from torch.utils.data import Dataset

# Mel spectrogram parameters
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 128
DURATION = 1.0  # seconds
N_FRAMES = int(SAMPLE_RATE * DURATION // HOP_LENGTH)  # ~125 frames

# ── Audio synthesis ──────────────────────────────────────────────

def _mel_filterbank(n_mels, n_fft, sr):
    """Simple mel filterbank (no librosa/torchaudio needed)."""
    n_freqs = n_fft // 2 + 1
    mel_min = 0.0
    mel_max = 2595.0 * math.log10(1.0 + sr / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]
        for f in range(left, center):
            fb[m - 1, f] = (f - left) / (center - left)
        for f in range(center, right):
            fb[m - 1, f] = (right - f) / (right - center)
    return torch.tensor(fb, dtype=torch.float32)


# Precompute mel filterbank
MEL_FB = _mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE)


def waveform_to_mel(waveform, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convert [T] waveform to [1, n_mels, T'] mel spectrogram."""
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    # STFT
    window = torch.hann_window(n_fft)
    spec = torch.stft(waveform, n_fft, hop_length, window=window,
                      return_complex=True)  # [F, T]
    mag = spec.abs()  # [F, T]

    # Mel
    n_freqs = n_fft // 2 + 1
    mel_spec = MEL_FB @ mag[:n_freqs, :]  # [n_mels, T]

    # Log compression
    mel_spec = torch.clamp(mel_spec, min=1e-10)
    mel_spec = torch.log(mel_spec)
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

    return mel_spec.unsqueeze(0)  # [1, n_mels, T]


def _sine_wave(freq, duration=DURATION, sr=SAMPLE_RATE, amplitude=0.5):
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * math.pi * freq * t)


def _square_wave(freq, duration=DURATION, sr=SAMPLE_RATE, amplitude=0.5):
    """Generate a square wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sign(np.sin(2 * math.pi * freq * t))


def _white_noise(duration=DURATION, sr=SAMPLE_RATE, amplitude=0.3):
    """Generate white noise."""
    return amplitude * np.random.uniform(-1, 1, int(sr * duration))


def _amplitude_modulated(freq_carrier=440, freq_mod=4, duration=DURATION,
                         sr=SAMPLE_RATE, amplitude=0.5):
    """Generate an amplitude-modulated tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    carrier = np.sin(2 * math.pi * freq_carrier * t)
    modulator = 0.5 * (1 + np.sin(2 * math.pi * freq_mod * t))
    return amplitude * carrier * modulator


def _frequency_modulated(freq_base=440, freq_dev=100, freq_mod=3,
                         duration=DURATION, sr=SAMPLE_RATE, amplitude=0.5):
    """Generate a frequency-modulated tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * math.pi * freq_base * t +
                              freq_dev / freq_mod * np.sin(2 * math.pi * freq_mod * t))


# ── Caption templates ───────────────────────────────────────────

_AUDIO_CAPTIONS = []

# Pure tones
for freq, note in [(220, 'low'), (440, 'medium'), (880, 'high'), (1760, 'very high')]:
    _AUDIO_CAPTIONS.append((lambda f=freq, n=note: (
        _sine_wave(f),
        f"A {n}-pitched pure tone" if n != 'low' else "A deep low-pitched tone"
    )))
    _AUDIO_CAPTIONS.append((lambda f=freq, n=note: (
        _sine_wave(f, amplitude=0.3),
        f"A quiet {n}-pitched tone"
    )))

# Square waves
for freq, note in [(110, 'low'), (220, 'buzzing'), (440, 'harsh')]:
    _AUDIO_CAPTIONS.append((lambda f=freq, n=note: (
        _square_wave(f),
        f"A {n} square wave sound"
    )))

# Noise
_AUDIO_CAPTIONS.append((lambda: (
    _white_noise(),
    "Static white noise"
)))
_AUDIO_CAPTIONS.append((lambda: (
    _white_noise(amplitude=0.1),
    "Faint background static"
)))

# AM tones
for freq, rate in [(440, 4), (880, 8), (220, 2)]:
    _AUDIO_CAPTIONS.append((lambda f=freq, r=rate: (
        _amplitude_modulated(freq_carrier=f, freq_mod=r),
        f"A pulsing tone at {f} Hz"
    )))

# FM tones
for base, dev in [(440, 200), (880, 300), (220, 100)]:
    _AUDIO_CAPTIONS.append((lambda b=base, d=dev: (
        _frequency_modulated(freq_base=b, freq_dev=d),
        f"A warbling tone"
    )))

# Multi-tone
_AUDIO_CAPTIONS.append((lambda: (
    _sine_wave(440) + 0.5 * _sine_wave(880),
    "A chord of two tones"
)))
_AUDIO_CAPTIONS.append((lambda: (
    _sine_wave(440) + _square_wave(220),
    "A mixed tone with harmonics"
)))


def generate_audio_sample(seed=None):
    """Generate (mel_tensor[1,128,~125], caption_str)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    idx = random.randrange(len(_AUDIO_CAPTIONS))
    waveform, caption = _AUDIO_CAPTIONS[idx]()

    # Normalize
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak * 0.5

    mel = waveform_to_mel(waveform)  # [1, n_mels, T]

    # Pad or truncate to N_FRAMES
    target_frames = N_FRAMES
    if mel.shape[-1] < target_frames:
        mel = torch.nn.functional.pad(mel, (0, target_frames - mel.shape[-1]))
    else:
        mel = mel[:, :, :target_frames]

    # Normalize to [-1, 1] range (matches image normalization)
    mel = mel.clamp(-3, 3) / 3.0

    return mel, caption


class AudioDataset(Dataset):
    """Synthetic audio dataset: (mel_spec, caption) pairs."""

    def __init__(self, num_samples=5000, seed=None):
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mel, caption = generate_audio_sample(seed=(self.seed or 0) + idx)
        return mel, caption


def get_audio_tokenizer_description():
    """Return a description of the audio dataset for training logs."""
    return f"AudioDataset: {len(_AUDIO_CAPTIONS)} templates, {N_FRAMES} frames, {N_MELS} mel bands"


if __name__ == "__main__":
    # Test
    mel, caption = generate_audio_sample(seed=42)
    print(f"Mel shape: {mel.shape}")  # [1, 128, ~125]
    print(f"Caption: {caption}")
    print(f"Mel range: [{mel.min():.3f}, {mel.max():.3f}]")

    ds = AudioDataset(num_samples=3)
    for i in range(3):
        m, c = ds[i]
        print(f"[{i}] {m.shape} - {c}")
    print("Audio synthetic data: OK")
