# UAV Tracker

A real-time UAV detection and tracking system using Python, OpenCV, YOLO, and ByteTrack.

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd uav-tracker
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the model

Place the UAV detection model in:

```text
models/drone-yolo26m.pt
```

### 5. Add a video

Place a test video in the `videos/` directory and update the video path in `main.py` if needed.

### 6. Run

```bash
python main.py
```

Press `Q` to stop the tracker.

## Current Features

* UAV detection with YOLO
* Multi-object tracking with ByteTrack
* Persistent track IDs
* Trajectory visualization
* Pixel-based motion telemetry
* Stale track cleanup

## Note

Motion is currently measured in pixels and pixels per second. Additional features and telemetry capabilities are still in development.
