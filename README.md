# VocaRig

VocaRig, sesten ARKit uyumlu dudak ve çene blendshape değerleri üreten bir eğitim ve çalışma zamanı projesidir. FaceRig ifade kanallarını yönetirken VocaRig konuşma artikülasyonuna odaklanır: ağız, dudak ve çene hareketleri ayrı bir akışta üretilir.

## Öne Çıkanlar

- Ses girdisinden 21 dudak/çene kanalına ve 52 ARKit blendshape çıktısına dönüşüm
- Streaming kullanım için GRU tabanlı dudak senkron modeli
- FastAPI tabanlı VocaRig Lab arayüzü
- Sentetik veri üretimi ve BEAT tabanlı işlenmiş veri hazırlama
- Eğitim, checkpoint, metrik takibi ve ONNX export akışı
- CPU/CUDA cihaz seçimi ve `fp32` / CUDA için `fp16_amp` eğitim desteği

## Proje Yapısı

```text
configs/                 Eğitim ve export ayarları
data/audio/              Varsayılan ses örnekleri
data/mesh/               ARKit avatar/mesh varlıkları
data/processed/          BEAT gibi işlenmiş veri setleri
data/synthetic/          Üretilmiş sentetik veri setleri
models/trained/          Final model, ONNX ve metrik çıktıları
models/training_checkpoints/  Eğitim sırasında ara checkpoint dosyaları
scripts/                 Repo içinden çalıştırılan yardımcı komutlar
src/vocarig/             Ana Python paketi
tests/                   Birim ve API testleri
```

## Kurulum

Python sürümü `3.14+` olmalı.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

## Hızlı Başlangıç

VocaRig Lab arayüzünü başlat:

```powershell
.\.venv\Scripts\python.exe scripts\run_ui.py
```

Tarayıcıdan aç:

```text
http://127.0.0.1:8010
```

Arayüzden model seçilebilir, ses dosyasıyla inference yapılabilir, gerçek zamanlı deneme çalıştırılabilir, veri seti seçilebilir, eğitim başlatılabilir ve ONNX export alınabilir.

## Temel İş Akışı

```text
ses -> özellik çıkarımı -> streaming GRU -> ARKit blendshape frame'leri
```

Offline ses dosyaları da aynı streaming hattı kullanır. Fark sadece verinin gerçek zamandan hızlı beslenmesidir.

## Veri Hazırlama

Sentetik veri üret:

```powershell
.\.venv\Scripts\python.exe scripts\generate_synthetic_data.py
```

BEAT tabanlı yaklaşık 1 saatlik VocaRig veri seti hazırla:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_beat_vocarig.py --force
```

Varsayılan çıktılar:

```text
data/synthetic/synthetic_vocarig.npz
data/processed/beat_vocarig_1h.npz
```

Eğitim arayüzü `data/synthetic` ve `data/processed` altındaki `.npz` dosyalarını otomatik listeler.

## Eğitim

Model eğit:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py
```

Belirli veri setiyle eğit:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --data data/processed/beat_vocarig_1h.npz
```

Önemli eğitim ayarları [configs/train_config.yaml](configs/train_config.yaml) içindedir:

- epoch, batch size, learning rate ve validation split
- sequence window/stride
- teacher forcing schedule
- pose, velocity, jerk, silence ve range loss ağırlıkları
- erken durdurma, divergence ve plateau kontrolleri
- checkpoint ve metrik yolları

Final çıktılar zaman damgalı olarak `models/trained/` altına yazılır:

```text
vocarig_lipsync_gru_YYYYMMDD-HHMMSS.pt
vocarig_lipsync_gru_YYYYMMDD-HHMMSS.onnx
vocarig_lipsync_gru_YYYYMMDD-HHMMSS_training_metrics.json
```

## ONNX Export

Seçili ya da config içindeki checkpoint'i ONNX formatına aktar:

```powershell
.\.venv\Scripts\python.exe scripts\export_onnx.py
```

Belirli checkpoint için:

```powershell
.\.venv\Scripts\python.exe scripts\export_onnx.py --checkpoint models/trained/vocarig_lipsync_gru_YYYYMMDD-HHMMSS.pt
```

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## API ve Arayüz

VocaRig Lab başlıca şu uçları kullanır:

- `GET /api/status`
- `GET /api/datasets`
- `GET /api/models`
- `POST /api/models/select`
- `POST /api/infer`
- `POST /api/infer/audio`
- `POST /api/infer/realtime`
- `POST /api/train/start`
- `POST /api/export`
- `POST /api/benchmark`

Statik varlıklar:

- `/mesh/ARKitMesh.glb`
- `/audio/voice-sample.wav`

## Lisans

Lisans bilgisi için [LICENSE](LICENSE) dosyasına bakın.
