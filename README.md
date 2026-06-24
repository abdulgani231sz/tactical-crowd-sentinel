# 🛡️ Tactical Crowd Sentinel v2

> **Real-time AI-powered crowd monitoring and threat detection system**  
> Built with YOLOv8 · ByteTrack · Bird's Eye View · Heatmap Overlay

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-green?style=flat-square&logo=opencv)
![CUDA](https://img.shields.io/badge/CUDA-RTX%20Ready-76B900?style=flat-square&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📸 Demo

![Crowd Sentinel Demo](demo.png)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **ByteTrack Tracker** | Multi-object tracking with Kalman filter — no ID switches on overlap |
| 🗺️ **Bird's Eye View Radar** | Perspective transform maps foot positions to top-down ground view |
| 🔥 **Live Heatmap** | Gaussian density accumulation overlay with temporal decay |
| ⚠️ **Advanced Threat Engine** | Running, crowd surge, group formation, counter-flow detection |
| 📊 **THREATCON Levels** | WHITE → ALPHA → BRAVO → CHARLIE escalation system |
| 🧵 **Threaded Inference** | Non-blocking GPU inference — draw loop never waits |
| 📝 **Event Log** | Timestamped alert history panel |
| 🏃 **Trail Rendering** | Per-person motion trails with color coding |

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/tactical-crowd-sentinel.git
cd tactical-crowd-sentinel
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python numpy scipy filterpy
```

### 3. Run
```bash
# Demo mode (no camera needed)
python tactical_crowd_sentinel_v2.py

# Webcam
python tactical_crowd_sentinel_v2.py --source 0

# Video file
python tactical_crowd_sentinel_v2.py --source crowd.mp4

# RTX GPU — best quality
python tactical_crowd_sentinel_v2.py --source crowd.mp4 --model yolov8s.pt --imgsz 640
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `Q` | Quit |

---

## ⚙️ Configuration

All settings in the `Cfg` dataclass at the top of the file:

```python
model:       "yolov8n.pt"   # n=fast, s=balanced, m=accurate
imgsz:       416             # inference resolution (416/640)
conf:        0.35            # detection confidence threshold
infer_every: 2               # infer every N frames (higher = faster display)
speed_run:   15.0            # px/frame threshold for "running" detection
density_hi:  12              # people count for HIGH density alert
group_min:   4               # min people to form a group cluster
```

---

## 🧠 How It Works

```
Video Frame
    │
    ▼
YOLOv8 (GPU Thread) ──► Person Detections [x1,y1,x2,y2,conf]
                              │
                              ▼
                      ByteTrack + Kalman ──► Confirmed Tracks [ID, bbox, velocity]
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
               Heatmap    Threat     BEV Radar
               Overlay    Engine     (Perspective
                            │         Transform)
                            ▼
                     THREATCON Level
                   WHITE/ALPHA/BRAVO/CHARLIE
                            │
                            ▼
                    Composite HUD Output
```

---

## 🔍 Threat Detection Logic

| Threat | Method |
|---|---|
| **Running** | Track speed > 15 px/frame threshold |
| **Crowd Surge** | 35%+ count spike in 6-frame rolling window |
| **Group Formation** | DBSCAN-style cluster detection (radius=85px, min=4) |
| **Counter Flow** | Velocity vector angle spread > 120° |
| **High Density** | People count > configurable threshold |

---

## 📦 Tech Stack

- **Detection** — YOLOv8n/s/m (Ultralytics)
- **Tracking** — Custom ByteTrack + filterpy Kalman Filter
- **Vision** — OpenCV 4.9+
- **Numerics** — NumPy, SciPy
- **Threading** — Python `threading` + `queue` (non-blocking inference)

---

## 📁 Project Structure

```
tactical-crowd-sentinel/
├── tactical_crowd_sentinel_v2.py   # Main application
├── requirements.txt                 # Dependencies
├── README.md                        # This file
└── demo.png                         # Screenshot
```

---

## 🖥️ Performance (RTX 3050)

| Config | FPS |
|---|---|
| yolov8n + imgsz=416 | ~30–45 FPS |
| yolov8s + imgsz=416 | ~20–30 FPS |
| yolov8s + imgsz=640 | ~15–22 FPS |

---

## 📄 License

MIT License — free to use for research, education, and portfolio purposes.

---

## 👤 Author

**Abdul Gani** —  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn]https://www.linkedin.com/in/abdul-gani-08sz/
