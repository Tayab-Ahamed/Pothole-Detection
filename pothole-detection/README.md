# 🛣️ Pothole Detection — YOLOv8

Real-time pothole detection powered by **YOLOv8** (Ultralytics).  
Supports live webcam feed, video files, and image folders — via a modern web dashboard or a CLI tool.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Live detection** | MJPEG webcam stream with bounding-box overlays |
| **File processing** | Upload images or videos; get annotated results back |
| **Background jobs** | Videos are processed in a background thread; poll for progress |
| **Severity scoring** | Potholes classified as Low / Medium / High by bounding-box area |
| **CSV logging** | Every detection logged with timestamp, coordinates, confidence, severity |
| **CLI mode** | Headless batch processing — webcam, video file, or image folder |
| **Custom training** | Fine-tune YOLOv8 on your own pothole dataset |
| **Download results** | Annotated images / videos downloadable from the dashboard |

---

## 📁 Project Structure

```
pothole-detection/
├── app.py                # Flask web application
├── detect_realtime.py    # CLI detection tool
├── train.py              # Model training script
├── utils.py              # Shared helpers (drawing, logging, severity)
├── data.yaml             # Dataset config template (edit before training)
├── requirements.txt
├── .gitignore
└── templates/
    └── index.html        # Web dashboard
```

> **Note:** `uploads/`, `outputs/`, and `runs/` are created automatically at runtime and are excluded from version control.  
> Model weights (`*.pt`) are also excluded — see [Model Weights](#-model-weights) below.

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- A webcam (for live feed)
- CUDA-capable GPU (optional but recommended for video processing)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/pothole-detection.git
cd pothole-detection

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Model Weights

The repository does **not** include model weights (`.pt` files) to keep it lightweight.

### Option A — Smoke-test with the base YOLOv8 model

`app.py` and `detect_realtime.py` automatically fall back to `yolov8n.pt` if `best_pothole.pt` is not found.  
Ultralytics will download `yolov8n.pt` on first run. This generic model is useful only to check the inference pipeline; it is not trained to detect potholes. Detection requires weights whose class labels explicitly include `pothole`. Generic class 0 (for example, `person`) is never treated as a pothole.

### Option B — Train a custom pothole model (recommended)

See [Training](#-training-a-custom-model) below.

### Option C — Download a pre-trained pothole model

Community models are available on [Roboflow Universe](https://universe.roboflow.com/search?q=pothole) and [Hugging Face](https://huggingface.co/models?search=pothole).  
Save your chosen model as `best_pothole.pt` in the project root.

---

## 🌐 Web Dashboard

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### Dashboard controls

| Control | Description |
|---|---|
| **Start / Stop** | Toggle live webcam detection |
| **Confidence slider** | Adjust detection threshold (0.10 – 0.95) |
| **Upload file** | Process an image or video file |
| **Download** | Save the annotated output |
| **Reset** | Clear the session detection counter |
| **Detection Log tab** | View processed files history |

---

## 💻 CLI Tool

```bash
python detect_realtime.py [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index, video file path, or image folder |
| `--model` | `yolov8n.pt` | Path to YOLOv8 `.pt` model file |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` (Apple Silicon) |
| `--conf` | `0.3` | Confidence threshold |
| `--save` | off | Save annotated output to `outputs/` |
| `--save-frames` | off | Save individual frames to `outputs/frames/` |
| `--log` | off | Write CSV log to `outputs/detections.csv` |
| `--show-fps` | off | Overlay FPS counter on video |

### Examples

```bash
# Webcam with FPS overlay
python detect_realtime.py --show-fps

# Video file on GPU, save result, log detections
python detect_realtime.py \
  --source road_footage.mp4 \
  --model best_pothole.pt \
  --device cuda \
  --conf 0.3 \
  --save --log

# Batch process an image folder
python detect_realtime.py \
  --source images/ \
  --model best_pothole.pt \
  --save --log

# Apple Silicon (M1/M2/M3)
python detect_realtime.py --device mps
```

Press **`q`** to quit the webcam / video window.

---

## 🏋️ Training a Custom Model

### 1. Prepare a dataset

Download a pothole dataset in **YOLOv8 format** from:
- [Roboflow Universe](https://universe.roboflow.com/search?q=pothole) — export as *YOLOv8*
- [Kaggle](https://www.kaggle.com/search?q=pothole+detection)

Place the exported data under `data/`:

```
data/
├── train/images/   ← training images
├── train/labels/   ← YOLO .txt annotations
├── val/images/
└── val/labels/
```

### 2. Edit `data.yaml`

The provided `data.yaml` is pre-configured for the layout above.  
Adjust `path`, `train`, `val` if your dataset is structured differently.

### 3. Train

```bash
python train.py \
  --model yolov8s.pt \
  --epochs 100 \
  --batch 16 \
  --device 0          # GPU 0; omit for CPU
```

| Model | Speed | Accuracy |
|---|---|---|
| `yolov8n.pt` | Fastest | Good |
| `yolov8s.pt` | Fast | Better ✓ |
| `yolov8m.pt` | Moderate | Great |
| `yolov8l.pt` | Slow | Excellent |

### 4. Use the trained model

```bash
cp runs/detect/pothole_detector/weights/best.pt best_pothole.pt
```

`app.py` and `detect_realtime.py` will automatically pick up `best_pothole.pt`.

---

## 📊 Detection Log CSV

When `--log` is used (CLI) or file processing runs, detections are appended to `outputs/detections.csv`:

| Column | Description |
|---|---|
| `timestamp` | ISO-8601 datetime |
| `frame_ref` | Frame index or image filename |
| `label` | `pothole(Low)` / `pothole(Medium)` / `pothole(High)` |
| `confidence` | Score in [0, 1] |
| `x1 y1 x2 y2` | Bounding-box pixel coordinates |
| `width height` | Bounding-box dimensions |
| `area_px` | Bounding-box area in pixels |

---

## 🤝 Contributing

1. Fork the repo and create a feature branch: `git checkout -b feat/my-feature`
2. Commit with clear messages: `git commit -m "feat: add severity heat-map overlay"`
3. Push and open a Pull Request

Bug reports and dataset contributions are welcome!

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
