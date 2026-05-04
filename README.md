# Smart Workplace Mask Compliance Monitor 🛡️

A real-time workplace safety system that detects whether individuals are
following face-mask compliance rules using **YOLOv8 object detection**,
**PyTorch**, and **OpenCV** — and streams live results to a dark-mode web
dashboard.

---

## ✨ Features

| Feature | Detail |
|---|---|
| Object detection | YOLOv8 (Ultralytics) — nano / small / medium / large |
| 3 classes | `with_mask` · `without_mask` · `mask_weared_incorrect` |
| Evaluation | mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS |
| Real-time | Webcam · video file · image directory |
| Dashboard | Dark-mode HTML/JS with live gauge, timeline, alert banner |
| Dataset format | Pascal-VOC XML → YOLO TXT (auto-converted) |

---

## 📂 Project Structure

```
mask-detection-system/
├── explore_dataset.py      # Step 1 — annotate & visualise the raw dataset
├── prepare_dataset.py      # Step 2 — convert VOC XML → YOLO TXT labels
├── train_yolo.py           # Step 3 — train YOLOv8 object detector
├── evaluate.py             # Step 4 — mAP@0.5, Precision, Recall, FPS
├── predict.py              # Step 5 — image / video / webcam inference
├── compliance_server.py    # Step 6 — Flask server (MJPEG + JSON API)
├── make_plots.py           # Generate training & evaluation plots
├── dataset.yaml            # YOLO dataset config (auto-written)
├── requirements.txt
├── dashboard/
│   └── index.html          # Live compliance web dashboard
├── data/
│   └── yolo/               # Prepared YOLO dataset (auto-created)
├── artifacts/
│   ├── yolo/               # YOLO training runs & best weights
│   ├── evaluation/         # mAP results & confusion matrix
│   ├── exploration/        # Dataset visualisation plots
│   └── plots/              # Training curve plots
└── src/mask_detection/     # Shared Python modules
    ├── constants.py
    └── dataset_tools.py
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> PyTorch CPU is installed automatically with `ultralytics`. For GPU support,
> install the matching CUDA wheel from [pytorch.org](https://pytorch.org/) first.

### 2. Explore the dataset

```bash
python explore_dataset.py
# ➜ artifacts/exploration/sample_grid.png
# ➜ artifacts/exploration/bbox_histogram.png
# ➜ artifacts/exploration/class_balance.png
# ➜ artifacts/exploration/dataset_stats.json
```

### 3. Prepare the YOLO dataset

```bash
python prepare_dataset.py --mode yolo
# ➜ data/yolo/{train,val,test}/{images,labels}/
# ➜ dataset.yaml  (updated with absolute paths)
```

### 4. Train YOLOv8

```bash
# Fast (nano backbone, ~10 min on CPU):
python train_yolo.py --model yolov8n.pt --epochs 50

# More accurate (small backbone):
python train_yolo.py --model yolov8s.pt --epochs 100 --batch 16

# ➜ artifacts/yolo/train/weights/best.pt
```

### 5. Evaluate (mAP@0.5)

```bash
python evaluate.py
# ➜ Prints mAP@0.5, Precision, Recall, FPS per class
# ➜ artifacts/evaluation/evaluation_results.json
# ➜ artifacts/evaluation/confusion_matrix.png (etc.)
```

### 6. Test on webcam / video / image

```bash
# Webcam (press Q to quit)
python predict.py --webcam

# Video file
python predict.py --video path/to/video.mp4

# Single image
python predict.py --image path/to/image.jpg

# Folder of images
python predict.py --input-dir path/to/folder/
```

### 7. Live compliance dashboard

```bash
python compliance_server.py
# ➜ Open http://localhost:5050/ in your browser
```

The dashboard shows:
- 📹 Live MJPEG video feed with detection overlays
- 🟢 **With Mask** / 🔴 **No Mask** / 🟡 **Incorrect** counters
- 🎯 Compliance % gauge chart
- 📈 60-second timeline chart
- ⚠️ Alert banner when violation rate exceeds 30 %

### 8. Generate training plots

```bash
python make_plots.py
# ➜ artifacts/plots/yolo_losses.png
# ➜ artifacts/plots/yolo_metrics.png
# ➜ artifacts/plots/per_class_metrics.png
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **mAP@0.5** | Mean Average Precision at IoU = 0.5 (primary metric) |
| **mAP@0.5:0.95** | mAP averaged over IoU thresholds 0.5→0.95 |
| **Precision** | Of all predicted positives, how many are correct |
| **Recall** | Of all true positives, how many were found |
| **FPS** | Frames processed per second (inference only) |

---

## ⚙️ Configuration

### Train with a different backbone

| Flag | Options | Notes |
|---|---|---|
| `--model` | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` | n = fastest, l = most accurate |
| `--epochs` | integer | 50 for quick test, 150+ for best results |
| `--imgsz` | 416, 640 | 640 is default |
| `--batch` | integer | 8–32 depending on RAM |
| `--device` | `cpu` `0` `mps` | empty = auto-detect |

### Dataset path

Edit `src/mask_detection/constants.py`:

```python
DEFAULT_RAW_DATASET_ROOT = "/Volumes/HP USB20FD/Smart Workplace Mask Compliance Monitor 2"
```

---

## 🗂️ Dataset Format

Raw dataset expected layout:
```
<dataset_root>/
  images/          ← JPEG/PNG images
  annotations/     ← Pascal-VOC XML files (one per image)
```

Classes: `with_mask`, `without_mask`, `mask_weared_incorrect`

---

## 📦 Tech Stack

- **[Ultralytics YOLOv8](https://docs.ultralytics.com/)** — object detection
- **PyTorch** — deep learning backend
- **OpenCV** — real-time frame processing
- **Flask** — compliance server & MJPEG streaming
- **Chart.js** — live dashboard charts
- **matplotlib / seaborn** — training plots

---

## 📜 License

MIT — free to use and modify for academic and commercial projects.
