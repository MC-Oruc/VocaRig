from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vocarig.audio.features import AudioFeatureConfig
from vocarig.models.blendshapes import LIP_BLENDSHAPE_NAMES, LIP_INDEX
from vocarig.synthetic.dataset import SyntheticGenerationConfig, generate_dataset
from vocarig.synthetic.visemes import phoneme_to_viseme, viseme_target


class SyntheticGenerationTests(unittest.TestCase):
    def test_synthetic_dataset_shapes_and_seed(self) -> None:
        config = SyntheticGenerationConfig(
            utterance_count=4,
            seed=7,
            min_phonemes=8,
            max_phonemes=10,
            audio=AudioFeatureConfig(context_frames=11, n_mels=80, fps=30),
        )
        first = generate_dataset(config)
        second = generate_dataset(config)

        self.assertEqual(first.audio_windows.shape[1:], (11, 80))
        self.assertEqual(first.previous_lip.shape[1], len(LIP_BLENDSHAPE_NAMES))
        self.assertEqual(first.y.shape[1], len(LIP_BLENDSHAPE_NAMES))
        self.assertTrue(np.array_equal(first.audio_windows, second.audio_windows))
        self.assertTrue(np.array_equal(first.y, second.y))
        self.assertTrue(np.all(first.y >= 0.0))
        self.assertTrue(np.all(first.y <= 1.0))

    def test_viseme_silence_keeps_jaw_near_zero(self) -> None:
        target = viseme_target(phoneme_to_viseme("sil"))
        self.assertLessEqual(float(target[LIP_INDEX["jawOpen"]]), 0.05)
        self.assertGreater(float(target[LIP_INDEX["mouthClose"]]), 0.0)


if __name__ == "__main__":
    unittest.main()
