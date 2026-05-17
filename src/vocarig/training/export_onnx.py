"""Export trained VocaRig checkpoints to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn
import yaml

from vocarig.models.lipsync_gru import LipSyncGRU
from vocarig.training.artifacts import onnx_path_for_checkpoint


class _OnnxLipSyncWrapper(nn.Module):
    def __init__(self, model: LipSyncGRU) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        audio_window: torch.Tensor,
        previous_lip: torch.Tensor,
        time_style: torch.Tensor,
        hidden_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.model(
            audio_window,
            previous_lip,
            time_style,
            hidden_in,
            steps_since_reset=self.model.warmup_steps,
        )
        return result.lip, result.hidden


def main() -> None:
    args = _parse_args()
    data = _load_yaml(args.config)
    export_config = data.get("export", {})
    checkpoint_path = Path(args.checkpoint or export_config.get("checkpoint_path", "models/vocarig_lipsync_gru.pt"))
    output_path = Path(args.output) if args.output else onnx_path_for_checkpoint(checkpoint_path)
    opset = int(args.opset or export_config.get("opset_version", 18))
    export_onnx(checkpoint_path, output_path, opset)


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    opset_version: int = 18,
) -> Path:
    """Export a VocaRig checkpoint to FP32 ONNX."""

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    if model_config.get("model_type") != "lipsync_gru":
        raise ValueError("Checkpoint is not a VocaRig LipSyncGRU model")
    model = LipSyncGRU(**{key: value for key, value in model_config.items() if key != "model_type"})
    model.load_state_dict(checkpoint["model_state_dict"])
    model.float().eval()
    wrapper = _OnnxLipSyncWrapper(model).eval()

    dummy_audio = torch.zeros(1, model.audio_context_frames, model.n_mels, dtype=torch.float32)
    dummy_previous = torch.zeros(1, model.lip_size, dtype=torch.float32)
    dummy_time_style = torch.zeros(1, model.time_size + model.style_size, dtype=torch.float32)
    dummy_hidden = torch.zeros(1, model.hidden_size, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_audio, dummy_previous, dummy_time_style, dummy_hidden),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["audio_window", "previous_lip", "time_style", "hidden_in"],
        output_names=["lip_output", "hidden_out"],
        dynamic_axes={
            "audio_window": {0: "batch"},
            "previous_lip": {0: "batch"},
            "time_style": {0: "batch"},
            "hidden_in": {0: "batch"},
            "lip_output": {0: "batch"},
            "hidden_out": {0: "batch"},
        },
        dynamo=False,
    )
    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)
    print(f"ONNX written: {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VocaRig LipSyncGRU to ONNX.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=None)
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


if __name__ == "__main__":
    main()
