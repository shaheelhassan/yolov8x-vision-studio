# YOLOv8x AI Vision Studio

A web-based computer vision detector powered by the Ultralytics YOLOv8x model (68.2M parameters), FastAPI, and shadcn/ui components.

## Features

- Local inference using yolov8x.pt (COCO mAP 53.9)
- GPU acceleration via CUDA with automatic CPU fallback
- Multi-task support: Object Detection, Instance Segmentation, Pose Estimation, Classification, and Oriented Bounding Box (OBB)
- 80 COCO classes categorized into 12 structured groups
- Interactive sidebar UI with confidence and IoU threshold sliders
- Live detection inspector with crop previews, category breakdown, and JSON export
- Webcam snapshot capture and built-in sample images

## Architecture

- Backend: Python FastAPI, Uvicorn, Ultralytics YOLO, PyTorch
- Frontend: React, Vite, Tailwind CSS, Radix UI primitives, shadcn/ui components
- Fonts: Poppins, Material Symbols

## Requirements

- Python 3.10 or higher
- Node.js 18 or higher (optional, pre-built static bundle included)
- NVIDIA GPU with CUDA drivers (optional, runs on CPU if unavailable)

Python packages:
- fastapi
- uvicorn
- ultralytics
- torch
- torchvision
- pillow

## Installation

1. Clone or navigate to the project directory:
```bash
cd "MINI DEX"
```

2. Install Python dependencies:
```bash
pip install fastapi uvicorn ultralytics torch torchvision pillow
```

3. Ensure yolov8x.pt is placed in the project root directory.

## Running the Application

### Option 1: Windows Batch File
Double-click start.bat or run from terminal:
```bash
start.bat
```

### Option 2: Python Command
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

## API Reference

### GET /api/info
Returns model metadata, hardware status (CUDA/CPU), supported tasks, and COCO taxonomy.

### POST /api/detect
Executes inference on an uploaded image or sample name.

Parameters:
- file: Image file (multipart/form-data)
- sample_name: Name of preset sample image (optional)
- task: detect | segment | pose | classify | obb
- conf_threshold: Float between 0.05 and 0.95
- iou_threshold: Float between 0.10 and 0.90
- classes_filter: Comma-separated list of class names (optional)
- device: auto | cuda | cpu

Response:
- total_detections: Integer
- inference_time_ms: Float
- fps: Float
- detections: Array of detected entities with bounding boxes, confidence, and crops
- annotated_image: Base64 data URI of annotated output

### GET /api/samples
Returns list of built-in sample images for testing.

## Frontend Development

To modify the frontend:
```bash
cd frontend
npm install
npm run dev
```

To build production assets:
```bash
npm run build
```
The build artifacts are output to static_dist/ and served directly by FastAPI.
