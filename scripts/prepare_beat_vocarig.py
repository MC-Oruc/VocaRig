"""Download a capped BEAT subset and convert it to the VocaRig NPZ schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vocarig.audio.features import AudioFeatureConfig, audio_to_feature_windows, load_wav
from vocarig.models.blendshapes import ARKIT_BLENDSHAPE_NAMES, LIP_BLENDSHAPE_NAMES


HF_DATASET = "H-Liu1997/BEAT"
BEAT_ROOT = "beat_english_v0.2.1/beat_english_v0.2.1"
RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
API_BASE = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main"
TARGET_SECONDS_DEFAULT = 3600.0
TARGET_TOLERANCE_SECONDS = 180.0
MAX_SECONDS = TARGET_SECONDS_DEFAULT + TARGET_TOLERANCE_SECONDS
MIN_SECONDS = TARGET_SECONDS_DEFAULT - TARGET_TOLERANCE_SECONDS
DEFAULT_HARD_SILENCE_ENERGY = 0.01
DEFAULT_SYNTHETIC_SILENCE_RATIO = 0.03
DEFAULT_SYNTHETIC_SILENCE_MIN_SECONDS = 0.5
DEFAULT_SYNTHETIC_SILENCE_MAX_SECONDS = 2.0
SILENCE_AUDIT_LIP_NAMES = (
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthShrugUpper",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthStretchLeft",
    "mouthStretchRight",
)


@dataclass(frozen=True)
class BeatClip:
    speaker: int
    stem: str
    wav_path: str
    json_path: str
    wav_size: int
    estimated_seconds: float


@dataclass(frozen=True)
class PrepareConfig:
    target_seconds: float = TARGET_SECONDS_DEFAULT
    max_seconds: float = MAX_SECONDS
    output_path: Path = ROOT / "data" / "processed" / "beat_vocarig_1h.npz"
    metadata_path: Path = ROOT / "data" / "processed" / "beat_vocarig_1h_metadata.json"
    raw_dir: Path = ROOT / "data" / "raw" / "beat_vocarig_1h"
    start_speaker: int = 1
    end_speaker: int = 30
    fps: int = 30
    sample_rate: int = 16_000
    force: bool = False
    seed: int = 42
    hard_silence_energy: float = DEFAULT_HARD_SILENCE_ENERGY
    synthetic_silence_ratio: float = DEFAULT_SYNTHETIC_SILENCE_RATIO
    synthetic_silence_min_seconds: float = DEFAULT_SYNTHETIC_SILENCE_MIN_SECONDS
    synthetic_silence_max_seconds: float = DEFAULT_SYNTHETIC_SILENCE_MAX_SECONDS


def main() -> None:
    args = _parse_args()
    config = PrepareConfig(
        target_seconds=float(args.target_seconds),
        max_seconds=float(args.max_seconds),
        output_path=Path(args.output),
        metadata_path=Path(args.metadata),
        raw_dir=Path(args.raw_dir),
        start_speaker=int(args.start_speaker),
        end_speaker=int(args.end_speaker),
        force=bool(args.force),
        seed=int(args.seed),
        hard_silence_energy=float(args.hard_silence_energy),
        synthetic_silence_ratio=float(args.synthetic_silence_ratio),
        synthetic_silence_min_seconds=float(args.synthetic_silence_min_seconds),
        synthetic_silence_max_seconds=float(args.synthetic_silence_max_seconds),
    )
    prepare_dataset(config)


def prepare_dataset(config: PrepareConfig) -> None:
    if config.max_seconds > 3900.0:
        raise ValueError("Refusing to download more than about 1 hour of BEAT data")
    if config.output_path.exists() and not config.force:
        raise FileExistsError(f"Output already exists: {config.output_path}")

    clips = select_clips(config)
    selected_seconds = sum(clip.estimated_seconds for clip in clips)
    if not MIN_SECONDS <= selected_seconds <= config.max_seconds:
        raise RuntimeError(
            f"Could not select about 1 hour without exceeding cap: {selected_seconds:.1f}s"
        )

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = download_clips(clips, config.raw_dir)
    dataset, silence_audit = convert_clips(downloaded, config)
    save_dataset(dataset, clips, config, silence_audit)


def select_clips(config: PrepareConfig) -> list[BeatClip]:
    selected: list[BeatClip] = []
    selected_seconds = 0.0
    for speaker in range(config.start_speaker, config.end_speaker + 1):
        for clip in list_speaker_clips(speaker):
            if selected_seconds + clip.estimated_seconds > config.max_seconds:
                if selected_seconds >= MIN_SECONDS:
                    return selected
                continue
            selected.append(clip)
            selected_seconds += clip.estimated_seconds
            if selected_seconds >= config.target_seconds:
                return selected
    raise RuntimeError(f"Only found {selected_seconds:.1f}s before source listing ended")


def list_speaker_clips(speaker: int) -> list[BeatClip]:
    prefix = f"{BEAT_ROOT}/{speaker}"
    items = list_tree(prefix)
    by_stem: dict[str, dict[str, dict]] = {}
    for item in items:
        path = str(item.get("path", ""))
        if not path:
            continue
        stem, suffix = _split_supported_file(path)
        if stem is None:
            continue
        by_stem.setdefault(stem, {})[suffix] = item

    clips: list[BeatClip] = []
    for stem, files in sorted(by_stem.items()):
        wav = files.get("wav")
        js = files.get("json")
        if not wav or not js:
            continue
        wav_size = int(wav.get("size") or 0)
        if wav_size <= 44:
            continue
        clips.append(
            BeatClip(
                speaker=speaker,
                stem=stem,
                wav_path=str(wav["path"]),
                json_path=str(js["path"]),
                wav_size=wav_size,
                estimated_seconds=max(0.0, (wav_size - 44) / 32000.0),
            )
        )
    return clips


def list_tree(prefix: str) -> list[dict]:
    escaped = "/".join(urllib.parse.quote(part) for part in prefix.split("/"))
    url = f"{API_BASE}/{escaped}?recursive=true&expand=true&limit=50"
    items: list[dict] = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            items.extend(json.load(response))
            link = response.headers.get("Link") or ""
        match = re.search(r'<([^>]+)>; rel="next"', link)
        url = match.group(1) if match else ""
    return items


def download_clips(clips: list[BeatClip], raw_dir: Path) -> list[tuple[BeatClip, Path, Path]]:
    result: list[tuple[BeatClip, Path, Path]] = []
    for index, clip in enumerate(clips, start=1):
        wav_path = raw_dir / clip.wav_path
        json_path = raw_dir / clip.json_path
        _download_file(clip.wav_path, wav_path)
        _download_file(clip.json_path, json_path)
        result.append((clip, wav_path, json_path))
        print(f"{index:03d}/{len(clips):03d} downloaded {clip.stem}")
    return result


def convert_clips(
    downloaded: list[tuple[BeatClip, Path, Path]],
    config: PrepareConfig,
) -> tuple[dict[str, np.ndarray], dict]:
    audio_config = AudioFeatureConfig(
        sample_rate=config.sample_rate,
        fps=config.fps,
        context_frames=11,
        n_mels=80,
    )
    lip_indices = [ARKIT_BLENDSHAPE_NAMES.index(name) for name in LIP_BLENDSHAPE_NAMES]
    audio_windows_rows: list[np.ndarray] = []
    previous_rows: list[np.ndarray] = []
    time_rows: list[np.ndarray] = []
    style_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    utterance_rows: list[np.ndarray] = []
    frame_rows: list[np.ndarray] = []
    speaker_rows: list[np.ndarray] = []
    audio_chunks: list[np.ndarray] = []
    audio_offsets: list[int] = []
    audio_lengths: list[int] = []
    sample_offset = 0
    real_frame_count = 0
    cleaned_hard_silence_frames = 0
    low_energy_lip_before_rows: list[np.ndarray] = []

    for utterance_id, (clip, wav_path, json_path) in enumerate(downloaded):
        samples, sample_rate = load_wav(str(wav_path))
        windows, times, energies = audio_to_feature_windows(samples, sample_rate, audio_config)
        arkit = load_arkit_sequence(json_path, times)
        frame_count = min(windows.shape[0], energies.shape[0], arkit.shape[0])
        windows = windows[:frame_count].astype(np.float32, copy=True)
        energies = energies[:frame_count].astype(np.float32, copy=True)
        lip = arkit[:frame_count, lip_indices].astype(np.float32, copy=True)
        hard_silence = energies <= config.hard_silence_energy
        if hard_silence.any():
            low_energy_lip_before_rows.append(lip[hard_silence].copy())
            windows[hard_silence] = 0.0
            lip[hard_silence] = 0.0
            energies[hard_silence] = 0.0
            cleaned_hard_silence_frames += int(hard_silence.sum())
        previous = _previous_from_lip(lip)
        real_frame_count += frame_count

        time_values = np.zeros((lip.shape[0], 3), dtype=np.float32)
        time_values[:, 0] = 1.0 / float(config.fps)
        time_values[:, 1] = _time_since_motion_change(lip, config.fps)
        time_values[:, 2] = energies
        style_values = np.repeat(np.asarray([[0.5, 0.5]], dtype=np.float32), lip.shape[0], axis=0)

        audio_windows_rows.append(windows)
        previous_rows.append(previous)
        time_rows.append(time_values)
        style_rows.append(style_values)
        y_rows.append(lip)
        utterance_rows.append(np.full(lip.shape[0], utterance_id, dtype=np.int32))
        frame_rows.append(np.arange(lip.shape[0], dtype=np.int32))
        speaker_rows.append(np.full(lip.shape[0], clip.speaker, dtype=np.int32))
        audio_offsets.append(sample_offset)
        audio_lengths.append(samples.size)
        audio_chunks.append(samples)
        sample_offset += samples.size

    rng = np.random.default_rng(config.seed)
    synthetic_silence_frames = 0
    synthetic_silence_sequences = 0
    target_silence_frames = max(0, int(round(real_frame_count * config.synthetic_silence_ratio)))
    min_silence_frames = max(1, int(round(config.synthetic_silence_min_seconds * config.fps)))
    max_silence_frames = max(min_silence_frames, int(round(config.synthetic_silence_max_seconds * config.fps)))
    while synthetic_silence_frames < target_silence_frames:
        frames = int(rng.integers(min_silence_frames, max_silence_frames + 1))
        utterance_id = len(downloaded) + synthetic_silence_sequences
        audio_windows_rows.append(np.zeros((frames, audio_config.context_frames, audio_config.n_mels), dtype=np.float32))
        previous_rows.append(np.zeros((frames, len(LIP_BLENDSHAPE_NAMES)), dtype=np.float32))
        time_values = np.zeros((frames, 3), dtype=np.float32)
        time_values[:, 0] = 1.0 / float(config.fps)
        time_values[:, 1] = np.minimum(np.arange(frames, dtype=np.float32) / float(config.fps), 1.0)
        time_rows.append(time_values)
        style_rows.append(np.repeat(np.asarray([[0.5, 0.5]], dtype=np.float32), frames, axis=0))
        y_rows.append(np.zeros((frames, len(LIP_BLENDSHAPE_NAMES)), dtype=np.float32))
        utterance_rows.append(np.full(frames, utterance_id, dtype=np.int32))
        frame_rows.append(np.arange(frames, dtype=np.int32))
        speaker_rows.append(np.zeros(frames, dtype=np.int32))
        audio_offsets.append(sample_offset)
        audio_length = max(1, int(round(frames / float(config.fps) * config.sample_rate)))
        audio_lengths.append(audio_length)
        audio_chunks.append(np.zeros(audio_length, dtype=np.float32))
        sample_offset += audio_length
        synthetic_silence_frames += frames
        synthetic_silence_sequences += 1

    dataset = {
        "audio_windows": np.concatenate(audio_windows_rows).astype(np.float32),
        "previous_lip": np.concatenate(previous_rows).astype(np.float32),
        "time_values": np.concatenate(time_rows).astype(np.float32),
        "style_values": np.concatenate(style_rows).astype(np.float32),
        "y": np.concatenate(y_rows).astype(np.float32),
        "utterance_ids": np.concatenate(utterance_rows).astype(np.int32),
        "frame_ids": np.concatenate(frame_rows).astype(np.int32),
        "speaker_ids": np.concatenate(speaker_rows).astype(np.int32),
        "audio_samples": np.concatenate(audio_chunks).astype(np.float32),
        "audio_offsets": np.asarray(audio_offsets, dtype=np.int64),
        "audio_lengths": np.asarray(audio_lengths, dtype=np.int64),
        "lip_names": np.asarray(LIP_BLENDSHAPE_NAMES),
        "arkit_names": np.asarray(ARKIT_BLENDSHAPE_NAMES),
    }
    energy = dataset["time_values"][:, 2]
    low_energy_after = energy <= config.hard_silence_energy
    zero_audio_windows = np.abs(dataset["audio_windows"]).sum(axis=(1, 2)) < 1e-8
    low_energy_before = (
        np.concatenate(low_energy_lip_before_rows, axis=0)
        if low_energy_lip_before_rows
        else np.zeros((0, len(LIP_BLENDSHAPE_NAMES)), dtype=np.float32)
    )
    silence_audit = {
        "hard_silence_energy": config.hard_silence_energy,
        "real_frame_count": int(real_frame_count),
        "cleaned_hard_silence_frames": int(cleaned_hard_silence_frames),
        "synthetic_silence_sequence_count": int(synthetic_silence_sequences),
        "synthetic_silence_frames": int(synthetic_silence_frames),
        "total_frames": int(dataset["y"].shape[0]),
        "zero_audio_window_count": int(zero_audio_windows.sum()),
        "zero_audio_window_ratio": float(zero_audio_windows.mean()),
        "low_energy_frame_count_after": int(low_energy_after.sum()),
        "low_energy_lip_before": _lip_summary(low_energy_before),
        "low_energy_lip_after": _lip_summary(dataset["y"][low_energy_after]),
    }
    return dataset, silence_audit


def load_arkit_sequence(json_path: Path, target_times: np.ndarray) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    source_names = [str(name) for name in data["names"]]
    frames = data["frames"]
    source_times = np.asarray([float(frame["time"]) for frame in frames], dtype=np.float32)
    source_weights = np.asarray([frame["weights"] for frame in frames], dtype=np.float32)
    name_to_index = {name: index for index, name in enumerate(source_names)}
    result = np.zeros((target_times.shape[0], len(ARKIT_BLENDSHAPE_NAMES)), dtype=np.float32)
    for target_index, name in enumerate(ARKIT_BLENDSHAPE_NAMES):
        source_index = name_to_index.get(name)
        if source_index is None:
            continue
        result[:, target_index] = np.interp(
            target_times,
            source_times,
            source_weights[:, source_index],
        ).astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def save_dataset(
    dataset: dict[str, np.ndarray],
    clips: list[BeatClip],
    config: PrepareConfig,
    silence_audit: dict,
) -> None:
    np.savez_compressed(config.output_path, **dataset)
    total_frames = int(dataset["y"].shape[0])
    total_audio_samples = int(dataset["audio_lengths"].sum())
    metadata = {
        "source": "BEAT",
        "source_dataset": HF_DATASET,
        "target": "VocaRig",
        "created_by": "scripts/prepare_beat_vocarig.py",
        "config": {
            "target_seconds": config.target_seconds,
            "max_seconds": config.max_seconds,
            "start_speaker": config.start_speaker,
            "end_speaker": config.end_speaker,
            "fps": config.fps,
            "sample_rate": config.sample_rate,
            "force": config.force,
            "seed": config.seed,
            "hard_silence_energy": config.hard_silence_energy,
            "synthetic_silence_ratio": config.synthetic_silence_ratio,
            "synthetic_silence_min_seconds": config.synthetic_silence_min_seconds,
            "synthetic_silence_max_seconds": config.synthetic_silence_max_seconds,
            "output_path": str(config.output_path),
            "metadata_path": str(config.metadata_path),
            "raw_dir": str(config.raw_dir),
        },
        "clip_count": len(clips),
        "speakers": sorted({clip.speaker for clip in clips}),
        "total_frames": total_frames,
        "fps": config.fps,
        "duration_seconds_from_frames": total_frames / float(config.fps),
        "duration_seconds_from_audio": total_audio_samples / float(config.sample_rate),
        "downloaded_raw_bytes": sum(clip.wav_size for clip in clips),
        "arrays": {key: list(value.shape) for key, value in dataset.items() if hasattr(value, "shape")},
        "silence_audit": silence_audit,
        "lip_names": LIP_BLENDSHAPE_NAMES,
        "arkit_names": ARKIT_BLENDSHAPE_NAMES,
        "clips": [asdict(clip) for clip in clips],
        "notes": [
            "Only WAV and facial JSON files were downloaded.",
            "BVH/body motion was not downloaded.",
            "BEAT JSON has no tongueOut in sampled files; missing channels are zero-filled.",
            "Blendshape values are mapped by name, not by source index.",
        ],
    }
    config.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {config.output_path}")
    print(f"saved {config.metadata_path}")
    print(f"duration_seconds_from_audio={metadata['duration_seconds_from_audio']:.1f}")
    print(f"duration_seconds_from_frames={metadata['duration_seconds_from_frames']:.1f}")


def _download_file(repo_path: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RESOLVE_BASE}/{urllib.parse.quote(repo_path, safe='/')}"
    urllib.request.urlretrieve(url, output_path)


def _split_supported_file(path: str) -> tuple[str | None, str | None]:
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix not in {"wav", "json"}:
        return None, None
    return str(Path(path).with_suffix("")).replace("\\", "/"), suffix


def _time_since_motion_change(values: np.ndarray, fps: int) -> np.ndarray:
    result = np.zeros(values.shape[0], dtype=np.float32)
    last_change = 0
    if values.shape[0] <= 1:
        return result
    deltas = np.abs(np.diff(values, axis=0)).mean(axis=1)
    threshold = max(0.002, float(np.percentile(deltas, 70)) if deltas.size else 0.002)
    for frame in range(1, values.shape[0]):
        if deltas[frame - 1] >= threshold:
            last_change = frame
        result[frame] = min(1.0, (frame - last_change) / float(fps))
    return result


def _previous_from_lip(lip: np.ndarray) -> np.ndarray:
    previous = np.zeros_like(lip)
    if lip.shape[0] > 1:
        previous[1:] = lip[:-1]
    return previous


def _lip_summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "count": 0,
            "overall_mean": 0.0,
            "overall_max": 0.0,
            "controls": {},
        }
    controls: dict[str, dict[str, float]] = {}
    for name in SILENCE_AUDIT_LIP_NAMES:
        index = LIP_BLENDSHAPE_NAMES.index(name)
        column = values[:, index]
        controls[name] = {
            "mean": float(column.mean()),
            "max": float(column.max()),
        }
    return {
        "count": int(values.shape[0]),
        "overall_mean": float(values.mean()),
        "overall_max": float(values.max()),
        "controls": controls,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-seconds", type=float, default=TARGET_SECONDS_DEFAULT)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "beat_vocarig_1h.npz"))
    parser.add_argument(
        "--metadata",
        default=str(ROOT / "data" / "processed" / "beat_vocarig_1h_metadata.json"),
    )
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "beat_vocarig_1h"))
    parser.add_argument("--start-speaker", type=int, default=1)
    parser.add_argument("--end-speaker", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-silence-energy", type=float, default=DEFAULT_HARD_SILENCE_ENERGY)
    parser.add_argument("--synthetic-silence-ratio", type=float, default=DEFAULT_SYNTHETIC_SILENCE_RATIO)
    parser.add_argument("--synthetic-silence-min-seconds", type=float, default=DEFAULT_SYNTHETIC_SILENCE_MIN_SECONDS)
    parser.add_argument("--synthetic-silence-max-seconds", type=float, default=DEFAULT_SYNTHETIC_SILENCE_MAX_SECONDS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
