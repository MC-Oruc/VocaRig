# VocaRig

Audio-to-ARKit lip/jaw blendshape workspace.

VocaRig complements FaceRig. FaceRig owns expression channels. VocaRig owns speech
articulation channels.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\generate_synthetic_data.py
.\.venv\Scripts\python.exe scripts\train_model.py
.\.venv\Scripts\python.exe scripts\export_onnx.py
.\.venv\Scripts\python.exe scripts\run_ui.py
```

UI:

```text
http://127.0.0.1:8010
```

Core pipeline:

```text
audio source -> chunker -> feature extractor -> streaming GRU -> ARKit frames
```

Offline files use the same streaming pipeline, just fed faster than realtime.

## Precision

Default training precision is `fp32`.

CUDA training can opt into:

```yaml
training:
  precision: fp16_amp
```

`fp16_amp` requires CUDA. Checkpoints are stored as FP32.

## Training Controls

UI and YAML now expose FaceRig-style training controls:

- scheduler: teacher forcing final ratio, decay start, decay epochs
- loss shaping: warmup loss steps, pose/delta/velocity/jerk/silence/range weights
- run shape: sequence window/stride, validation split, checkpoint and metric intervals
- auto stop: target val/train loss, divergence, plateau, overfit gap
- synthetic data: utterances, seed, phoneme range, silence probability, TR/EN mix

Current defaults are final-training oriented, not smoke-test oriented:

- synthetic set: 4000 utterances, 18-56 phonemes, TR/EN 55/45
- training: 1800 epochs, batch 64, sequence 96/stride 24, fp32
- schedule: teacher forcing decays from epoch 250 over 700 epochs
- safety: divergence stop and plateau stop enabled
