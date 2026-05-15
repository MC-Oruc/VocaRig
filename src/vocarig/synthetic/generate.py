"""CLI for synthetic data generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vocarig.audio.features import AudioFeatureConfig
from vocarig.synthetic.dataset import SyntheticGenerationConfig, generate_dataset, save_dataset


def main() -> None:
    args = _parse_args()
    config_data = _load_yaml(args.config)
    audio_config = config_data.get("audio", {})
    synthetic_config = config_data.get("synthetic", {})
    data_config = config_data.get("data", {})

    audio = AudioFeatureConfig(
        sample_rate=int(audio_config.get("sample_rate", 16000)),
        n_mels=int(audio_config.get("n_mels", 80)),
        window_ms=float(audio_config.get("window_ms", 25.0)),
        hop_ms=float(audio_config.get("hop_ms", 10.0)),
        context_frames=int(audio_config.get("context_frames", 11)),
        fps=int(audio_config.get("fps", data_config.get("fps", 30))),
        f_min=float(audio_config.get("f_min", 50.0)),
        f_max=float(audio_config.get("f_max", 7600.0)),
    )
    language_mix = synthetic_config.get("language_mix", {}) or {}
    generation_config = SyntheticGenerationConfig(
        utterance_count=int(args.utterances or synthetic_config.get("utterance_count", 4000)),
        seed=int(args.seed if args.seed is not None else synthetic_config.get("seed", 42)),
        sample_rate=audio.sample_rate,
        fps=audio.fps,
        min_phonemes=int(synthetic_config.get("min_phonemes", 18)),
        max_phonemes=int(synthetic_config.get("max_phonemes", 56)),
        silence_probability=float(synthetic_config.get("silence_probability", 0.18)),
        tr_probability=float(language_mix.get("tr", 0.55)),
        audio=audio,
    )
    output_path = Path(args.output or synthetic_config.get("output_path", "data/synthetic/synthetic_vocarig.npz"))
    metadata_path = Path(
        args.metadata or synthetic_config.get("metadata_path", "data/synthetic/synthetic_vocarig_metadata.json")
    )

    dataset = generate_dataset(generation_config)
    save_dataset(dataset, output_path, metadata_path, generation_config)
    print(f"dataset: {output_path}")
    print(f"metadata: {metadata_path}")
    print(f"frames: {dataset.y.shape[0]}")
    print(f"audio_windows: {dataset.audio_windows.shape}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic VocaRig lip-sync data.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--utterances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metadata", default=None)
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


if __name__ == "__main__":
    main()
