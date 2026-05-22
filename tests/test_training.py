from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import onnx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vocarig.audio.features import AudioFeatureConfig
from vocarig.models.lipsync_gru import LipSyncGRU
from vocarig.models.blendshapes import LIP_INDEX
from vocarig.synthetic.dataset import SyntheticGenerationConfig, generate_dataset, save_dataset
from vocarig.training.data import load_sequence_npz_dataset, split_indices
from vocarig.training.export_onnx import export_onnx
from vocarig.training.precision import PrecisionConfig, build_precision_runtime
from vocarig.training.train import TrainingRunConfig, _sequence_loss_components, _teacher_forcing_ratio, train_model


class TrainingTests(unittest.TestCase):
    def test_model_shape_range_and_step_bound(self) -> None:
        model = LipSyncGRU(max_step=0.12, reference_dt=1 / 30)
        audio = torch.zeros(3, 11, 80)
        previous = torch.zeros(3, 21)
        time_style = torch.zeros(3, 5)
        time_style[:, 0] = 1 / 30
        output = model(audio, previous, time_style)

        self.assertEqual(tuple(output.lip.shape), (3, 21))
        self.assertEqual(tuple(output.hidden.shape), (3, 128))
        self.assertTrue(torch.all(output.lip >= 0.0))
        self.assertTrue(torch.all(output.lip <= 1.0))
        self.assertLessEqual(float(output.delta_effect.abs().max()), 0.120001)

    def test_fp16_amp_requires_cuda(self) -> None:
        with self.assertRaises(RuntimeError):
            build_precision_runtime(PrecisionConfig("fp16_amp"), torch.device("cpu"))

    def test_teacher_forcing_schedule(self) -> None:
        config = TrainingRunConfig(
            final_teacher_forcing_ratio=0.25,
            teacher_decay_start_epoch=3,
            teacher_decay_epochs=4,
        )
        self.assertEqual(_teacher_forcing_ratio(1, config), 1.0)
        self.assertAlmostEqual(_teacher_forcing_ratio(3, config), 0.8125)
        self.assertAlmostEqual(_teacher_forcing_ratio(6, config), 0.25)

    def test_grouped_split_keeps_utterances_out_of_both_sets(self) -> None:
        groups = np.repeat(np.arange(6), 3)
        train_indices, val_indices = split_indices(groups.shape[0], 0.33, 42, groups)

        train_groups = set(groups[train_indices.numpy()].tolist())
        val_groups = set(groups[val_indices.numpy()].tolist())
        self.assertTrue(train_groups)
        self.assertTrue(val_groups)
        self.assertFalse(train_groups & val_groups)

    def test_short_utterance_is_padded_into_sequence_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.npz"
            frames = 3
            np.savez_compressed(
                path,
                audio_windows=np.zeros((frames, 11, 80), dtype=np.float32),
                previous_lip=np.zeros((frames, 21), dtype=np.float32),
                time_values=np.zeros((frames, 3), dtype=np.float32),
                style_values=np.zeros((frames, 2), dtype=np.float32),
                y=np.zeros((frames, 21), dtype=np.float32),
                utterance_ids=np.zeros(frames, dtype=np.int32),
                frame_ids=np.arange(frames, dtype=np.int32),
            )

            bundle = load_sequence_npz_dataset(path, sequence_window=5, sequence_stride=5)

        self.assertEqual(len(bundle.dataset), 1)
        self.assertEqual(bundle.metadata["padded_short_sequence_count"], 1)
        self.assertEqual(bundle.y.shape[1], 5)

    def test_silence_loss_penalizes_all_lip_controls(self) -> None:
        lip = torch.zeros(21)
        lip[LIP_INDEX["mouthStretchRight"]] = 1.0
        model = _ConstantLipModel(lip)
        audio = torch.zeros(1, 2, 11, 80)
        previous = torch.zeros(1, 2, 21)
        time_style = torch.zeros(1, 2, 5)
        time_style[:, :, 0] = 1 / 30
        y = lip.view(1, 1, 21).expand(1, 2, 21).clone()

        losses = _sequence_loss_components(
            model,
            audio,
            previous,
            time_style,
            y,
            TrainingRunConfig(delta_loss_weight=0.0, warmup_loss_steps=0),
        )

        self.assertGreater(float(losses["silence_loss"]), 0.0)
        self.assertEqual(TrainingRunConfig().delta_loss_weight, 0.0)

    def test_mini_training_and_onnx_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "synthetic.npz"
            metadata_path = root / "synthetic_metadata.json"
            checkpoint_path = root / "vocarig.pt"
            metrics_path = root / "metrics.json"
            onnx_path = root / "vocarig.onnx"

            audio = AudioFeatureConfig(context_frames=11, n_mels=80, fps=30)
            synthetic_config = SyntheticGenerationConfig(
                utterance_count=6,
                seed=11,
                min_phonemes=12,
                max_phonemes=14,
                audio=audio,
            )
            dataset = generate_dataset(synthetic_config)
            save_dataset(dataset, data_path, metadata_path, synthetic_config)

            metrics = train_model(
                TrainingRunConfig(
                    data_path=data_path,
                    checkpoint_path=checkpoint_path,
                    metrics_path=metrics_path,
                    checkpoint_dir=root / "checkpoints",
                    epochs=2,
                    batch_size=2,
                    sequence_window=8,
                    sequence_stride=8,
                    hidden_size=32,
                    audio_channels=16,
                    device="cpu",
                    precision="fp32",
                    log_interval=1,
                    final_teacher_forcing_ratio=0.0,
                    teacher_decay_start_epoch=1,
                    teacher_decay_epochs=2,
                    warmup_loss_steps=2,
                    stop_on_divergence_loss=True,
                    divergence_loss=10.0,
                    pose_loss_weight=1.0,
                    delta_loss_weight=0.2,
                )
            )
            export_onnx(checkpoint_path, onnx_path)
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            checkpoint_exists = checkpoint_path.exists()
            metrics_exists = metrics_path.exists()
            onnx_exists = onnx_path.exists()

        self.assertEqual(metrics["completed_epochs"], 2)
        self.assertEqual(metrics["precision"], "fp32")
        self.assertFalse(metrics["amp_enabled"])
        self.assertIn("teacher_forcing_ratio", metrics["history"][-1])
        self.assertEqual(metrics["training_config"]["delta_loss_weight"], 0.2)
        self.assertFalse(metrics["early_stopped"])
        self.assertTrue(checkpoint_exists)
        self.assertTrue(metrics_exists)
        self.assertTrue(onnx_exists)


class _ConstantLipModel(torch.nn.Module):
    def __init__(self, lip: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("lip", lip.float())

    def initial_hidden(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, 1, device=device, dtype=dtype)

    def forward(
        self,
        audio_window: torch.Tensor,
        previous_lip: torch.Tensor,
        time_style: torch.Tensor,
        hidden: torch.Tensor | None = None,
        steps_since_reset: torch.Tensor | int | float | None = None,
    ) -> SimpleNamespace:
        lip = self.lip.to(device=audio_window.device, dtype=audio_window.dtype).expand(audio_window.shape[0], -1)
        next_hidden = torch.zeros(audio_window.shape[0], 1, device=audio_window.device, dtype=audio_window.dtype)
        return SimpleNamespace(lip=lip, hidden=next_hidden, raw_lip=lip)


if __name__ == "__main__":
    unittest.main()
