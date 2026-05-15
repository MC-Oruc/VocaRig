"""Causal audio chunking and log-mel feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
import io
import wave

import numpy as np


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16_000
    n_mels: int = 80
    window_ms: float = 25.0
    hop_ms: float = 10.0
    context_frames: int = 11
    fps: int = 30
    f_min: float = 50.0
    f_max: float = 7_600.0

    @property
    def window_size(self) -> int:
        return max(1, int(round(self.sample_rate * self.window_ms / 1000.0)))

    @property
    def hop_size(self) -> int:
        return max(1, int(round(self.sample_rate * self.hop_ms / 1000.0)))


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float32 samples."""

    with wave.open(path, "rb") as handle:
        return _read_wave_handle(handle)


def load_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Load WAV bytes as mono float32 samples."""

    with wave.open(io.BytesIO(data), "rb") as handle:
        return _read_wave_handle(handle)


def write_wav_bytes(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    """Encode mono float samples as 16-bit WAV bytes."""

    array = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (array * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def resample_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono samples with deterministic linear interpolation."""

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return mono
    if mono.size == 0:
        return mono
    duration = mono.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, mono.size, endpoint=False)
    target_x = np.linspace(0.0, duration, target_size, endpoint=False)
    return np.interp(target_x, source_x, mono).astype(np.float32)


def audio_to_feature_windows(
    samples: np.ndarray,
    sample_rate: int,
    config: AudioFeatureConfig,
    volume_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return streaming frame windows, frame times, and frame energies."""

    mono = resample_mono(samples, sample_rate, config.sample_rate)
    if mono.size == 0:
        mono = np.zeros(config.hop_size, dtype=np.float32)
    mel = log_mel_spectrogram(mono, config)
    frame_count = max(1, int(np.ceil((mono.size / config.sample_rate) * config.fps)))
    context = config.context_frames
    windows = np.zeros((frame_count, context, config.n_mels), dtype=np.float32)
    times = np.arange(frame_count, dtype=np.float32) / float(config.fps)
    energies = np.zeros(frame_count, dtype=np.float32)
    threshold = float(np.clip(volume_threshold, 0.0, 1.0))

    for frame_index, time_s in enumerate(times):
        mel_index = int(np.floor(time_s / (config.hop_ms / 1000.0)))
        start = mel_index - context + 1
        for offset in range(context):
            source = start + offset
            if 0 <= source < mel.shape[0]:
                windows[frame_index, offset] = mel[source]
        sample_end = min(mono.size, int(round((time_s + 1.0 / config.fps) * config.sample_rate)))
        sample_start = max(0, sample_end - int(round(0.08 * config.sample_rate)))
        if sample_end > sample_start:
            rms = float(np.sqrt(np.mean(np.square(mono[sample_start:sample_end]))))
            if rms < threshold:
                windows[frame_index] = 0.0
                energies[frame_index] = 0.0
            else:
                energies[frame_index] = np.clip(rms * 8.0, 0.0, 1.0)

    return windows, times, energies.astype(np.float32)


def log_mel_spectrogram(samples: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    """Compute log-mel features with only NumPy."""

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    window = config.window_size
    hop = config.hop_size
    if mono.size < window:
        mono = np.pad(mono, (0, window - mono.size))
    frame_count = 1 + max(0, (mono.size - window) // hop)
    frames = np.lib.stride_tricks.sliding_window_view(mono, window)[::hop][:frame_count]
    if frames.size == 0:
        frames = np.zeros((1, window), dtype=np.float32)
    taper = np.hanning(window).astype(np.float32)
    spectrum = np.fft.rfft(frames * taper, axis=1)
    power = np.square(np.abs(spectrum)).astype(np.float32)
    filters = mel_filter_bank(
        n_fft=window,
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=min(config.f_max, config.sample_rate / 2.0),
    )
    mel = power @ filters.T
    log_mel = np.log1p(mel * 10.0)
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, keepdims=True) + 1e-5
    return ((log_mel - mean) / std).astype(np.float32)


def mel_filter_bank(
    n_fft: int,
    sample_rate: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Create a triangular mel filter bank."""

    fft_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, fft_bins - 1)
    filters = np.zeros((n_mels, fft_bins), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bin_points[index : index + 3]
        if center <= left:
            center = min(left + 1, fft_bins - 1)
        if right <= center:
            right = min(center + 1, fft_bins - 1)
        for bin_index in range(left, center):
            filters[index, bin_index] = (bin_index - left) / max(1, center - left)
        for bin_index in range(center, right):
            filters[index, bin_index] = (right - bin_index) / max(1, right - center)
    return filters


def _read_wave_handle(handle: wave.Wave_read) -> tuple[np.ndarray, int]:
    channels = handle.getnchannels()
    sample_width = handle.getsampwidth()
    sample_rate = handle.getframerate()
    frames = handle.readframes(handle.getnframes())
    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), sample_rate


def _hz_to_mel(value: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (np.power(10.0, np.asarray(value) / 2595.0) - 1.0)
