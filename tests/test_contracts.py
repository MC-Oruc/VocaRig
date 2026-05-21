from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vocarig.models.blendshapes import ARKIT_BLENDSHAPE_NAMES, LIP_BLENDSHAPE_NAMES, LIP_INDEX
from vocarig.runtime.mixer import merge_face_and_lip


class ContractTests(unittest.TestCase):
    def test_arkit_and_lip_contract_sizes(self) -> None:
        self.assertEqual(len(ARKIT_BLENDSHAPE_NAMES), 52)
        self.assertEqual(len(set(ARKIT_BLENDSHAPE_NAMES)), 52)
        self.assertEqual(len(LIP_BLENDSHAPE_NAMES), 21)
        self.assertEqual(len(set(LIP_BLENDSHAPE_NAMES)), 21)
        self.assertIn("jawOpen", LIP_INDEX)
        self.assertIn("mouthPucker", LIP_INDEX)

    def test_merge_face_and_lip_policy(self) -> None:
        lip = np.zeros(len(LIP_BLENDSHAPE_NAMES), dtype=np.float32)
        lip[LIP_INDEX["jawOpen"]] = 0.7
        lip[LIP_INDEX["mouthPressLeft"]] = 0.2
        merged = merge_face_and_lip({"mouthPressLeft": 0.8, "mouthSmileLeft": 0.6}, lip)

        self.assertAlmostEqual(merged["jawOpen"], 0.7)
        self.assertAlmostEqual(merged["mouthPressLeft"], 0.8)
        self.assertAlmostEqual(merged["mouthSmileLeft"], 0.6)
        self.assertEqual(len(merged), 52)


if __name__ == "__main__":
    unittest.main()
