from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.prepare_beat_vocarig import BeatClip, PrepareConfig, convert_clips
from vocarig.models.blendshapes import ARKIT_BLENDSHAPE_NAMES


class BeatSilenceCleanupTests(unittest.TestCase):
    def test_low_energy_frames_are_neutral_and_silence_sequences_are_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_path = root / "clip.wav"
            json_path = root / "clip.json"
            _write_silent_wav(wav_path, sample_rate=16_000, seconds=1.0)
            _write_open_mouth_json(json_path, fps=30, seconds=1.0)

            clip = BeatClip(
                speaker=1,
                stem="clip",
                wav_path="clip.wav",
                json_path="clip.json",
                wav_size=wav_path.stat().st_size,
                estimated_seconds=1.0,
            )
            config = PrepareConfig(
                output_path=root / "out.npz",
                metadata_path=root / "out_metadata.json",
                raw_dir=root,
                synthetic_silence_ratio=0.2,
                synthetic_silence_min_seconds=0.5,
                synthetic_silence_max_seconds=0.5,
            )

            dataset, audit = convert_clips([(clip, wav_path, json_path)], config)

        low_energy = dataset["time_values"][:, 2] <= config.hard_silence_energy
        self.assertGreater(int(low_energy.sum()), 0)
        self.assertTrue(np.allclose(dataset["audio_windows"][low_energy], 0.0))
        self.assertTrue(np.allclose(dataset["y"][low_energy], 0.0))
        self.assertGreater(audit["low_energy_lip_before"]["overall_max"], 0.0)
        self.assertEqual(audit["low_energy_lip_after"]["overall_max"], 0.0)
        self.assertGreaterEqual(audit["zero_audio_window_ratio"], 0.1)
        self.assertGreater(audit["synthetic_silence_sequence_count"], 0)

        for utterance_id in np.unique(dataset["utterance_ids"]):
            indices = np.flatnonzero(dataset["utterance_ids"] == utterance_id)
            self.assertTrue(np.allclose(dataset["previous_lip"][indices[0]], 0.0))
            if indices.size > 1:
                self.assertTrue(np.allclose(dataset["previous_lip"][indices[1:]], dataset["y"][indices[:-1]]))


def _write_silent_wav(path: Path, sample_rate: int, seconds: float) -> None:
    samples = np.zeros(int(round(sample_rate * seconds)), dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def _write_open_mouth_json(path: Path, fps: int, seconds: float) -> None:
    frame_count = int(round(fps * seconds))
    jaw_index = ARKIT_BLENDSHAPE_NAMES.index("jawOpen")
    stretch_index = ARKIT_BLENDSHAPE_NAMES.index("mouthStretchRight")
    lower_index = ARKIT_BLENDSHAPE_NAMES.index("mouthLowerDownRight")
    frames = []
    for frame in range(frame_count):
        weights = [0.0] * len(ARKIT_BLENDSHAPE_NAMES)
        weights[jaw_index] = 0.8
        weights[stretch_index] = 0.7
        weights[lower_index] = 0.6
        frames.append({"time": frame / float(fps), "weights": weights, "rotation": []})
    path.write_text(json.dumps({"names": ARKIT_BLENDSHAPE_NAMES, "frames": frames}), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
