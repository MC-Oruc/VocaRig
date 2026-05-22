from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vocarig.audio.features import write_wav_bytes
from vocarig.models.blendshapes import LIP_BLENDSHAPE_NAMES
from vocarig.models.lipsync_gru import LipSyncGRU
import vocarig.ui.app as ui_app


class UIApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_config_path = ui_app.CONFIG_PATH
        self.old_model_path = ui_app.app.state.model_path
        self.old_device = ui_app.app.state.device_mode
        model = LipSyncGRU()
        self.checkpoint_path = self.root / "vocarig.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model.config_dict(),
                "lip_names": LIP_BLENDSHAPE_NAMES,
            },
            self.checkpoint_path,
        )
        config_path = self.root / "train_config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": model.config_dict(),
                    "audio": {
                        "sample_rate": 16000,
                        "n_mels": 80,
                        "context_frames": 11,
                        "fps": 30,
                    },
                    "synthetic": {
                        "output_path": str(self.root / "synthetic.npz"),
                        "metadata_path": str(self.root / "synthetic_metadata.json"),
                    },
                    "training": {
                        "metrics_path": str(self.root / "metrics.json"),
                        "checkpoint_dir": str(self.root / "training_checkpoints"),
                        "precision": "fp32",
                    },
                    "export": {
                        "checkpoint_path": str(self.checkpoint_path),
                        "onnx_path": str(self.root / "vocarig.onnx"),
                        "opset_version": 18,
                    },
                }
            ),
            encoding="utf-8",
        )
        ui_app.CONFIG_PATH = config_path
        ui_app.app.state.model_path = None
        ui_app.app.state.device_mode = "cpu"
        self.client = TestClient(ui_app.app)

    def tearDown(self) -> None:
        ui_app.CONFIG_PATH = self.old_config_path
        ui_app.app.state.model_path = self.old_model_path
        ui_app.app.state.device_mode = self.old_device
        self.tmp.cleanup()

    def test_index_and_status(self) -> None:
        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn("VocaRig Lab", index.text)

        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["lip_names"]), 21)
        self.assertEqual(len(payload["arkit_names"]), 52)

    def test_infer_endpoint_and_websocket(self) -> None:
        payload = {
            "stream_id": "test",
            "previous_lip": [0.0] * 21,
            "delta_time": 1 / 30,
            "style_values": [0.5, 0.5],
        }
        response = self.client.post("/api/infer", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["lip_values"]), 21)
        self.assertEqual(len(data["arkit_values"]), 52)

        with self.client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json(payload)
            data = websocket.receive_json()
        self.assertTrue(data["ok"])
        self.assertEqual(len(data["lip_values"]), 21)

    def test_audio_file_uses_streaming_pipeline(self) -> None:
        wav = write_wav_bytes(torch.zeros(1600).numpy(), 16000)
        response = self.client.post(
            "/api/infer/audio",
            files={"file": ("test.wav", wav, "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(payload["frame_count"], 1)
        self.assertEqual(len(payload["frames"][0]["lip_values"]), 21)
        self.assertIn("infer_ms", payload)
        self.assertEqual(payload["volume_threshold"], 0.015)

    def test_realtime_audio_endpoint_accepts_pcm_chunks(self) -> None:
        request = {
            "stream_id": "live-test",
            "samples": [0.0] * 800,
            "sample_rate": 16000,
            "previous_lip": [0.0] * 21,
            "delta_time": 1 / 30,
            "style_values": [0.5, 0.5],
            "volume_threshold": 0.02,
            "reset_state": True,
        }
        response = self.client.post("/api/infer/realtime", json=request)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "realtime")
        self.assertEqual(len(payload["lip_values"]), 21)
        self.assertIn("latency_ms", payload)
        self.assertEqual(payload["volume_threshold"], 0.02)

        with self.client.websocket_connect("/ws/infer") as websocket:
            websocket.send_json({**request, "request_id": "rt-1"})
            data = websocket.receive_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["request_id"], "rt-1")
        self.assertEqual(data["mode"], "realtime")
        self.assertEqual(len(data["lip_values"]), 21)

    def test_device_and_missing_files_endpoints(self) -> None:
        response = self.client.post("/api/device", json={"device": "cpu"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["selected"], "cpu")

        self.assertEqual(self.client.get("/api/data").status_code, 200)
        self.assertEqual(self.client.get("/api/metrics").status_code, 200)

    def test_dataset_options_include_synthetic_and_processed(self) -> None:
        synthetic_dir = ui_app.ROOT / "data" / "synthetic"
        processed_dir = ui_app.ROOT / "data" / "processed"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        synthetic_path = synthetic_dir / "ui_test_synthetic.npz"
        processed_path = processed_dir / "ui_test_processed.npz"
        arrays = {
            "audio_windows": np.zeros((4, 11, 80), dtype=np.float32),
            "previous_lip": np.zeros((4, 21), dtype=np.float32),
            "time_values": np.zeros((4, 3), dtype=np.float32),
            "style_values": np.zeros((4, 2), dtype=np.float32),
            "y": np.zeros((4, 21), dtype=np.float32),
            "lip_names": np.asarray(LIP_BLENDSHAPE_NAMES),
        }
        np.savez(synthetic_path, **arrays)
        np.savez(processed_path, **arrays)
        self.addCleanup(lambda: synthetic_path.exists() and synthetic_path.unlink())
        self.addCleanup(lambda: processed_path.exists() and processed_path.unlink())

        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)
        kinds = {item["kind"] for item in response.json()["options"]}
        self.assertIn("synthetic", kinds)
        self.assertIn("processed", kinds)

        info = self.client.get("/api/data", params={"path": str(processed_path)})
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()["y_shape"], [4, 21])

    def test_synthetic_endpoint_accepts_advanced_options(self) -> None:
        response = self.client.post(
            "/api/synthetic/generate",
            json={
                "utterances": 1,
                "seed": 5,
                "min_phonemes": 2,
                "max_phonemes": 2,
                "silence_probability": 0.1,
                "tr_probability": 0.7,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        
        # Wait for async generation to finish
        if ui_app.synthetic_job.thread:
            ui_app.synthetic_job.thread.join()
            
        self.assertTrue((self.root / "synthetic.npz").exists())

    def test_training_request_accepts_advanced_options(self) -> None:
        request = ui_app.TrainingRequest(
            data_path="data/processed/beat_vocarig_1h.npz",
            epochs=2,
            batch_size=4,
            learning_rate=0.001,
            weight_decay=0.0002,
            sequence_window=16,
            final_teacher_forcing_ratio=0.25,
            stop_on_plateau=True,
            delta_loss_weight=0.2,
        )
        self.assertEqual(request.data_path, "data/processed/beat_vocarig_1h.npz")
        self.assertEqual(request.sequence_window, 16)
        self.assertTrue(request.stop_on_plateau)
        self.assertEqual(request.delta_loss_weight, 0.2)

    def test_bad_model_selection_rejected(self) -> None:
        response = self.client.post("/api/models/select", json={"path": str(self.root / "bad.pt")})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
