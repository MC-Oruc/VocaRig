"""Synthetic audio/lip dataset generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import concurrent.futures
from pathlib import Path
import tempfile
import os

import numpy as np

from vocarig.audio.features import AudioFeatureConfig, audio_to_feature_windows
from vocarig.models.blendshapes import ARKIT_BLENDSHAPE_NAMES, LIP_BLENDSHAPE_NAMES
from vocarig.synthetic.visemes import EN_PHONEMES, TR_PHONEMES, phoneme_to_viseme, viseme_target


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    utterance_count: int = 4000
    seed: int = 42
    sample_rate: int = 16_000
    fps: int = 30
    min_phonemes: int = 18
    max_phonemes: int = 56
    silence_probability: float = 0.18
    tr_probability: float = 0.55
    audio: AudioFeatureConfig = AudioFeatureConfig()


@dataclass(frozen=True)
class SyntheticDataset:
    audio_windows: np.ndarray
    previous_lip: np.ndarray
    time_values: np.ndarray
    style_values: np.ndarray
    y: np.ndarray
    utterance_ids: np.ndarray
    frame_ids: np.ndarray
    phoneme_ids: np.ndarray
    audio_samples: np.ndarray
    audio_offsets: np.ndarray
    audio_lengths: np.ndarray


def _process_utterance(
    utterance_id: int, 
    config: SyntheticGenerationConfig, 
    base_seed: int,
    tmp_path: str
) -> str:
    rng = np.random.default_rng(base_seed + utterance_id)
    style = np.asarray([rng.uniform(0.25, 0.95), rng.uniform(0.0, 1.0)], dtype=np.float32)
    language = "tr" if rng.random() < config.tr_probability else "en"
    phonemes, durations = _sample_utterance(language, rng, config)
    audio, timeline = _synthesize_audio(phonemes, durations, style, rng, config.sample_rate)

    windows, times, energies = audio_to_feature_windows(audio, config.sample_rate, config.audio)
    lip_targets, phoneme_ids = _render_lip_sequence(
        times,
        timeline,
        style,
        energies,
        rng,
        config.fps,
    )
    
    n_frames = windows.shape[0]
    
    previous = np.zeros((n_frames, len(LIP_BLENDSHAPE_NAMES)), dtype=np.float32)
    if n_frames > 0:
        previous[1:] = lip_targets[:-1]

    time_vals = np.zeros((n_frames, 3), dtype=np.float32)
    time_vals[:, 0] = 1.0 / float(config.fps)
    time_vals[:, 2] = energies

    tl_idx = 0
    tl_len = len(timeline)
    for frame_id in range(n_frames):
        time_s = float(times[frame_id])
        while tl_idx < tl_len - 1 and timeline[tl_idx]["end"] <= time_s:
            tl_idx += 1
        item_start = float(timeline[tl_idx]["start"])
        time_vals[frame_id, 1] = max(0.0, min(1.0, (time_s - item_start) / 0.5))

    style_vals = np.repeat(style[np.newaxis, :], n_frames, axis=0)
    u_ids = np.full(n_frames, utterance_id, dtype=np.int32)
    f_ids = np.arange(n_frames, dtype=np.int32)

    filename = str(Path(tmp_path) / f"out_{utterance_id}.npz")
    np.savez_compressed(
        filename, 
        windows=windows, 
        previous=previous, 
        time_vals=time_vals, 
        style_vals=style_vals, 
        lip_targets=lip_targets, 
        u_ids=u_ids, 
        f_ids=f_ids, 
        phoneme_ids=phoneme_ids, 
        audio=audio
    )
    return filename

def generate_dataset(config: SyntheticGenerationConfig, progress_callback=None) -> SyntheticDataset:
    """Generate a deterministic synthetic lip-sync dataset."""
    if config.utterance_count <= 0:
        raise ValueError("utterance_count must be greater than 0")
    if config.min_phonemes <= 0 or config.max_phonemes < config.min_phonemes:
        raise ValueError("invalid phoneme count range")
    
    sample_offset = 0

    import os
    workers = min(os.cpu_count() or 4, 16)
    
    results = [None] * config.utterance_count
    with tempfile.TemporaryDirectory() as tmp_path:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_utterance, uid, config, config.seed, tmp_path): uid
                for uid in range(config.utterance_count)
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                uid = futures[future]
                results[uid] = future.result()
                completed += 1
                if progress_callback:
                    progress_callback(completed, config.utterance_count, "generating")

        if progress_callback:
            progress_callback(config.utterance_count, config.utterance_count, "aggregating")

        # Ok, now we load stats from files
        shapes = []
        for filename in results:
            with np.load(filename, allow_pickle=False) as data:
                shapes.append((data["windows"].shape[0], data["audio"].size, data["windows"].shape[1:]))

        total_frames = sum(s[0] for s in shapes)
        total_audio_samples = sum(s[1] for s in shapes)

        if total_frames == 0:
            return SyntheticDataset(
                audio_windows=np.zeros((0, 0, 0), dtype=np.float32),
                previous_lip=np.zeros((0, len(LIP_BLENDSHAPE_NAMES)), dtype=np.float32),
                time_values=np.zeros((0, 3), dtype=np.float32),
                style_values=np.zeros((0, 2), dtype=np.float32),
                y=np.zeros((0, len(LIP_BLENDSHAPE_NAMES)), dtype=np.float32),
                utterance_ids=np.zeros((0,), dtype=np.int32),
                frame_ids=np.zeros((0,), dtype=np.int32),
                phoneme_ids=np.zeros((0,), dtype=np.int32),
                audio_samples=np.zeros((0,), dtype=np.float32),
                audio_offsets=np.zeros((0,), dtype=np.int64),
                audio_lengths=np.zeros((0,), dtype=np.int64),
            )

        audio_windows = np.empty((total_frames, *shapes[0][2]), dtype=np.float32)
        previous_lip = np.empty((total_frames, len(LIP_BLENDSHAPE_NAMES)), dtype=np.float32)
        time_values = np.empty((total_frames, 3), dtype=np.float32)
        style_values = np.empty((total_frames, 2), dtype=np.float32)
        y = np.empty((total_frames, len(LIP_BLENDSHAPE_NAMES)), dtype=np.float32)
        utterance_ids = np.empty((total_frames,), dtype=np.int32)
        frame_ids = np.empty((total_frames,), dtype=np.int32)
        phoneme_ids = np.empty((total_frames,), dtype=np.int32)
        
        audio_samples = np.empty((total_audio_samples,), dtype=np.float32)
        audio_offsets = np.empty((config.utterance_count,), dtype=np.int64)
        audio_lengths = np.empty((config.utterance_count,), dtype=np.int64)

        frame_offset = 0
        sample_offset = 0

        for i, filename in enumerate(results):
            with np.load(filename, allow_pickle=False) as data:
                frames = data["windows"].shape[0]
                a_len = data["audio"].size
                
                audio_windows[frame_offset:frame_offset+frames] = data["windows"]
                previous_lip[frame_offset:frame_offset+frames] = data["previous"]
                time_values[frame_offset:frame_offset+frames] = data["time_vals"]
                style_values[frame_offset:frame_offset+frames] = data["style_vals"]
                y[frame_offset:frame_offset+frames] = data["lip_targets"]
                utterance_ids[frame_offset:frame_offset+frames] = data["u_ids"]
                frame_ids[frame_offset:frame_offset+frames] = data["f_ids"]
                phoneme_ids[frame_offset:frame_offset+frames] = data["phoneme_ids"]
                
                audio_samples[sample_offset:sample_offset+a_len] = data["audio"]
                audio_offsets[i] = sample_offset
                audio_lengths[i] = a_len
                
            frame_offset += frames
            sample_offset += a_len

    if progress_callback:
        progress_callback(config.utterance_count, config.utterance_count, "saving")

    return SyntheticDataset(
        audio_windows=audio_windows,
        previous_lip=previous_lip,
        time_values=time_values,
        style_values=style_values,
        y=y,
        utterance_ids=utterance_ids,
        frame_ids=frame_ids,
        phoneme_ids=phoneme_ids,
        audio_samples=audio_samples,
        audio_offsets=audio_offsets,
        audio_lengths=audio_lengths,
    )


def save_dataset(
    dataset: SyntheticDataset,
    output_path: Path,
    metadata_path: Path,
    config: SyntheticGenerationConfig,
) -> None:
    """Save generated arrays and metadata."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        audio_windows=dataset.audio_windows,
        previous_lip=dataset.previous_lip,
        time_values=dataset.time_values,
        style_values=dataset.style_values,
        y=dataset.y,
        utterance_ids=dataset.utterance_ids,
        frame_ids=dataset.frame_ids,
        phoneme_ids=dataset.phoneme_ids,
        audio_samples=dataset.audio_samples,
        audio_offsets=dataset.audio_offsets,
        audio_lengths=dataset.audio_lengths,
        lip_names=np.asarray(LIP_BLENDSHAPE_NAMES),
        arkit_names=np.asarray(ARKIT_BLENDSHAPE_NAMES),
    )
    metadata = {
        "config": {
            **asdict(config),
            "audio": asdict(config.audio),
        },
        "audio_windows_shape": list(dataset.audio_windows.shape),
        "previous_lip_shape": list(dataset.previous_lip.shape),
        "time_values_shape": list(dataset.time_values.shape),
        "style_values_shape": list(dataset.style_values.shape),
        "y_shape": list(dataset.y.shape),
        "lip_names": LIP_BLENDSHAPE_NAMES,
        "arkit_names": ARKIT_BLENDSHAPE_NAMES,
        "schema": {
            "audio_windows": ["frame", "context", "mel"],
            "previous_lip": ["frame", "lip"],
            "time_values": ["delta_time", "time_since_phoneme_change", "energy"],
            "style_values": ["articulation", "asymmetry"],
            "y": ["frame", "lip"],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _sample_utterance(
    language: str,
    rng: np.random.Generator,
    config: SyntheticGenerationConfig,
) -> tuple[list[str], list[float]]:
    phones = TR_PHONEMES if language == "tr" else EN_PHONEMES
    count = int(rng.integers(config.min_phonemes, config.max_phonemes + 1))
    phonemes: list[str] = ["sil"]
    durations: list[float] = [float(rng.uniform(0.08, 0.24))]
    for _ in range(count):
        if rng.random() < config.silence_probability:
            phonemes.append("sil")
            durations.append(float(rng.uniform(0.05, 0.22)))
        phoneme = str(rng.choice(phones))
        phonemes.append(phoneme)
        if phoneme in {"a", "e", "i", "ii", "o", "oe", "u", "ue", "aa", "ae", "ah", "eh", "iy", "ih", "ow", "uw"}:
            duration = float(rng.uniform(0.09, 0.20))
        else:
            duration = float(rng.uniform(0.045, 0.12))
        durations.append(duration)
    phonemes.append("sil")
    durations.append(float(rng.uniform(0.10, 0.30)))
    return phonemes, durations


def _synthesize_audio(
    phonemes: list[str],
    durations: list[float],
    style: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int,
) -> tuple[np.ndarray, list[dict]]:
    chunks: list[np.ndarray] = []
    timeline: list[dict] = []
    t = 0.0
    phase = float(rng.uniform(0.0, np.pi * 2.0))
    for index, (phoneme, duration) in enumerate(zip(phonemes, durations, strict=True)):
        size = max(1, int(round(duration * sample_rate)))
        local_time = np.arange(size, dtype=np.float32) / float(sample_rate)
        viseme = phoneme_to_viseme(phoneme)
        base_freq = _phoneme_frequency(phoneme, viseme)
        envelope = np.sin(np.linspace(0.0, np.pi, size, dtype=np.float32))
        amplitude = 0.12 + 0.18 * float(style[0])
        if viseme == "sil":
            chunk = np.zeros(size, dtype=np.float32)
        elif viseme in {"s", "sh", "fv", "tdn", "kg"}:
            noise = rng.normal(0.0, 1.0, size=size).astype(np.float32)
            tone = np.sin(2.0 * np.pi * base_freq * local_time + phase).astype(np.float32)
            chunk = amplitude * envelope * (0.70 * noise + 0.30 * tone)
        else:
            tone1 = np.sin(2.0 * np.pi * base_freq * local_time + phase)
            tone2 = np.sin(2.0 * np.pi * base_freq * 2.05 * local_time + phase * 0.5)
            tone3 = np.sin(2.0 * np.pi * base_freq * 3.20 * local_time + phase * 0.25)
            chunk = amplitude * envelope * (0.70 * tone1 + 0.20 * tone2 + 0.10 * tone3)
        chunk += rng.normal(0.0, 0.002, size=size).astype(np.float32)
        chunk = np.clip(chunk, -0.95, 0.95).astype(np.float32)
        chunks.append(chunk)
        timeline.append(
            {
                "index": index,
                "phoneme": phoneme,
                "viseme": viseme,
                "start": t,
                "end": t + duration,
            }
        )
        t += duration
        phase += duration * base_freq * 2.0 * np.pi
    return np.concatenate(chunks).astype(np.float32), timeline


def _render_lip_sequence(
    times: np.ndarray,
    timeline: list[dict],
    style: np.ndarray,
    energies: np.ndarray,
    rng: np.random.Generator,
    fps: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = times.shape[0]
    targets = np.zeros((n_frames, len(LIP_BLENDSHAPE_NAMES)), dtype=np.float32)
    phoneme_ids = np.zeros(n_frames, dtype=np.int32)
    current = np.zeros(len(LIP_BLENDSHAPE_NAMES), dtype=np.float32)
    dt = 1.0 / float(fps)
    anticipation = 0.045
    release = 0.82
    
    tl_idx = 0
    tl_len = len(timeline)
    
    for frame_index in range(n_frames):
        time_s = float(times[frame_index])
        t_ant = time_s + anticipation
        
        while tl_idx < tl_len - 1 and timeline[tl_idx]["end"] <= t_ant:
            tl_idx += 1
            
        phoneme = timeline[tl_idx]
        phoneme_ids[frame_index] = int(phoneme["index"])
        
        target = viseme_target(str(phoneme["viseme"]), style)
        energy = float(energies[frame_index])
        if energy < 0.025:
            target *= 0.18
        target *= (0.82 + 0.35 * energy)
        
        speed = 9.5 if str(phoneme["viseme"]) in {"bmp", "sil"} else 7.0
        alpha = min(speed * dt, 1.0)
        
        current = current + (target - current) * alpha
        if energy < 0.02:
            current *= release
            
        current += rng.normal(0.0, 0.003, size=current.shape).astype(np.float32)
        targets[frame_index] = np.clip(current, 0.0, 1.0)
        
    return targets, phoneme_ids


def _phoneme_at(time_s: float, timeline: list[dict]) -> dict:
    for item in timeline:
        if float(item["start"]) <= time_s < float(item["end"]):
            return item
    return timeline[-1]


def _time_since_phoneme_change(time_s: float, timeline: list[dict]) -> float:
    item = _phoneme_at(time_s, timeline)
    return float(np.clip((time_s - float(item["start"])) / 0.5, 0.0, 1.0))


def _phoneme_frequency(phoneme: str, viseme: str) -> float:
    if viseme == "aa":
        return 180.0
    if viseme == "ee":
        return 260.0
    if viseme == "ih":
        return 300.0
    if viseme == "oh":
        return 150.0
    if viseme == "uu":
        return 130.0
    if viseme in {"s", "sh"}:
        return 4200.0
    if viseme in {"fv", "tdn", "kg"}:
        return 1800.0
    return 220.0 + (sum(ord(char) for char in phoneme) % 90)
