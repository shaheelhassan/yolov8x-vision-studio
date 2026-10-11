import os
import io
import time
import base64
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO

app = FastAPI(title="YOLOv8x AI Vision Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov8x.pt")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# 80 COCO Classes categorized into the exact 12 groups requested by user
COCO_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "People": {
        "icon": "person",
        "color": "#3b82f6",  # blue
        "classes": ["person"]
    },
    "Vehicles": {
        "icon": "directions_car",
        "color": "#06b6d4",  # cyan
        "classes": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
    },
    "Outdoor Objects": {
        "icon": "traffic",
        "color": "#f59e0b",  # amber
        "classes": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"]
    },
    "Animals": {
        "icon": "pets",
        "color": "#10b981",  # emerald
        "classes": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
    },
    "Accessories": {
        "icon": "work",
        "color": "#8b5cf6",  # purple
        "classes": ["backpack", "umbrella", "handbag", "tie", "suitcase"]
    },
    "Sports Equipment": {
        "icon": "sports_tennis",
        "color": "#ec4899",  # pink
        "classes": [
            "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"
        ]
    },
    "Kitchen Items": {
        "icon": "restaurant",
        "color": "#14b8a6",  # teal
        "classes": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"]
    },
    "Food": {
        "icon": "lunch_dining",
        "color": "#f97316",  # orange
        "classes": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]
    },
    "Furniture": {
        "icon": "chair",
        "color": "#6366f1",  # indigo
        "classes": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"]
    },
    "Electronics": {
        "icon": "devices",
        "color": "#0284c7",  # light blue
        "classes": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"]
    },
    "Appliances": {
        "icon": "kitchen",
        "color": "#64748b",  # slate
        "classes": ["microwave", "oven", "toaster", "sink", "refrigerator"]
    },
    "Indoor Objects": {
        "icon": "home",
        "color": "#a855f7",  # violet
        "classes": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    }
}

CLASS_TO_CATEGORY: Dict[str, Dict[str, str]] = {}
for cat_name, cat_data in COCO_CATEGORIES.items():
    for cls in cat_data["classes"]:
        CLASS_TO_CATEGORY[cls] = {
            "category": cat_name,
            "color": cat_data["color"],
            "icon": cat_data["icon"]
        }

POSE_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Model Cache
MODELS: Dict[str, YOLO] = {}

def get_model(task: str = "detect") -> YOLO:
    if task not in MODELS:
        if task == "detect":
            model_file = WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else "yolov8x.pt"
            print(f"Loading YOLOv8x Detection model from {model_file}...")
            MODELS[task] = YOLO(model_file)
        elif task == "segment":
            print("Loading YOLOv8x Segmentation model (yolov8x-seg.pt)...")
            MODELS[task] = YOLO("yolov8x-seg.pt")
        elif task == "pose":
            print("Loading YOLOv8x Pose model (yolov8x-pose.pt)...")
            MODELS[task] = YOLO("yolov8x-pose.pt")
        elif task == "classify":
            print("Loading YOLOv8x Classification model (yolov8x-cls.pt)...")
            MODELS[task] = YOLO("yolov8x-cls.pt")
        elif task == "obb":
            print("Loading YOLOv8x OBB model (yolov8x-obb.pt)...")
            MODELS[task] = YOLO("yolov8x-obb.pt")
        else:
            MODELS[task] = YOLO(WEIGHTS_PATH)
    return MODELS[task]

# Preload default detect model
try:
    get_model("detect")
except Exception as e:
    print(f"Initial model load notice: {e}")

@app.get("/api/info")
def get_system_info():
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Standard)"
    vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)) if cuda_available else 0

    return {
        "model": {
            "name": "YOLOv8x (Extra Large)",
            "family": "Ultralytics YOLOv8",
            "weights_file": "yolov8x.pt",
            "weights_found": os.path.exists(WEIGHTS_PATH),
            "file_size_mb": round(os.path.getsize(WEIGHTS_PATH) / (1024 * 1024), 1) if os.path.exists(WEIGHTS_PATH) else 0,
            "parameters": "68.2 Million",
            "default_resolution": "640x640",
            "coco_map_50_95": "53.9 mAP",
            "capabilities": {
                "tasks": [
                    {"id": "detect", "name": "Object Detection", "weights": "yolov8x.pt", "active": True, "description": "80 COCO classes with accurate bounding boxes"},
                    {"id": "segment", "name": "Instance Segmentation", "weights": "yolov8x-seg.pt", "active": False, "description": "Pixel-level object segmentation masks"},
                    {"id": "pose", "name": "Pose Estimation", "weights": "yolov8x-pose.pt", "active": False, "description": "17 human body keypoint tracking"},
                    {"id": "classify", "name": "Image Classification", "weights": "yolov8x-cls.pt", "active": False, "description": "Top-1 and Top-5 ImageNet category classification"},
                    {"id": "obb", "name": "Oriented Bounding Box (OBB)", "weights": "yolov8x-obb.pt", "active": False, "description": "Rotated bounding box detection for aerial & oblique views"}
                ],
                "coco_classes_count": 80,
                "categories": COCO_CATEGORIES,
                "pose_keypoints": POSE_KEYPOINTS,
                "custom_training_domains": [
                    "Defects in manufacturing",
                    "License plates & traffic control",
                    "Helmets & PPE safety gear",
                    "Faces & biometric detection",
                    "Retail shelf analytics",
                    "Medical imaging features",
                    "Crops, weeds, and pests in agriculture",
                    "Logos, signs, and text regions"
                ],
                "use_cases": [
                    "Surveillance and security",
                    "Traffic monitoring and counting",
                    "Retail analytics & heatmaps",
                    "Sports analytics & athlete tracking",
                    "Industrial quality inspection",
                    "Autonomous vehicles and robotics"
                ]
            }
        },
        "hardware": {
            "cuda_available": cuda_available,
            "device_name": device_name,
            "vram_mb": vram_mb,
            "torch_version": torch.__version__
        }
    }

@app.get("/api/samples")
def get_sample_images():
    samples = [
        {
            "id": "bus",
            "name": "Urban Traffic & Street",
            "file": "bus.jpg",
            "categories": ["Vehicles", "People"],
            "url": "/static/samples/bus.jpg"
        },
        {
            "id": "zidane",
            "name": "Sports Personalities",
            "file": "zidane.jpg",
            "categories": ["People", "Accessories"],
            "url": "/static/samples/zidane.jpg"
        },
        {
            "id": "animals",
            "name": "Playful Dogs & Pets",
            "file": "animals.jpg",
            "categories": ["Animals", "Outdoor Objects"],
            "url": "/static/samples/animals.jpg"
        },
        {
            "id": "kitchen",
            "name": "Modern Dining & Kitchen",
            "file": "kitchen.jpg",
            "categories": ["Furniture", "Kitchen Items", "Appliances"],
            "url": "/static/samples/kitchen.jpg"
        },
        {
            "id": "sports",
            "name": "Athletics & Action",
            "file": "sports.jpg",
            "categories": ["Sports Equipment", "People"],
            "url": "/static/samples/sports.jpg"
        }
    ]
    # Filter only existing files
    valid_samples = [s for s in samples if os.path.exists(os.path.join(SAMPLES_DIR, s["file"]))]
    return valid_samples

def image_to_base64(pil_image: Image.Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    if format.upper() == "PNG":
        pil_image.save(buffered, format="PNG")
        mime = "image/png"
    else:
        pil_image.save(buffered, format="JPEG", quality=90)
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/api/detect")
async def detect_image(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    task: str = Form("detect"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    classes_filter: Optional[str] = Form(None),
    device: str = Form("auto")
):
    try:
        # Load image
        if file is not None and file.filename:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        elif sample_name:
            sample_path = os.path.join(SAMPLES_DIR, sample_name)
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=404, detail=f"Sample '{sample_name}' not found")
            pil_img = Image.open(sample_path).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="No image file or sample provided")

        img_w, img_h = pil_img.size

        # Device selection
        if device == "auto":
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and torch.cuda.is_available():
            target_device = "cuda:0"
        else:
            target_device = "cpu"

        # Load appropriate YOLO model
        model = get_model(task)

        # Parse class filter
        selected_classes = None
        if classes_filter:
            requested = [c.strip().lower() for c in classes_filter.split(",") if c.strip()]
            if requested:
                # Map names to class IDs
                name_to_id = {v.lower(): k for k, v in model.names.items()}
                selected_classes = [name_to_id[name] for name in requested if name in name_to_id]

        start_time = time.perf_counter()

        # Run inference
        results = model.predict(
            source=pil_img,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=selected_classes,
            device=target_device,
            verbose=False
        )
        total_time = (time.perf_counter() - start_time) * 1000

        result = results[0]
        speed = result.speed
        inference_time_ms = round(speed.get("inference", total_time), 1)
        preprocess_time_ms = round(speed.get("preprocess", 0.0), 1)
        postprocess_time_ms = round(speed.get("postprocess", 0.0), 1)

        # Annotated image
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[..., ::-1]
        annotated_pil = Image.fromarray(annotated_rgb)
        annotated_base64 = image_to_base64(annotated_pil, format="JPEG")
        original_base64 = image_to_base64(pil_img, format="JPEG")

        detections = []
        category_counts: Dict[str, int] = {}
        class_counts: Dict[str, int] = {}

        # Parse boxes
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for idx in range(len(boxes)):
                box = boxes[idx]
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"Class {cls_id}")
                conf = float(box.conf[0].item())
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                x1, y1, x2, y2 = xyxy

                # Normalized coords
                nx1 = max(0.0, min(1.0, x1 / img_w))
                ny1 = max(0.0, min(1.0, y1 / img_h))
                nx2 = max(0.0, min(1.0, x2 / img_w))
                ny2 = max(0.0, min(1.0, y2 / img_h))
                nw = nx2 - nx1
                nh = ny2 - ny1

                cat_info = CLASS_TO_CATEGORY.get(cls_name.lower(), {
                    "category": "Other",
                    "color": "#64748b",
                    "icon": "label"
                })

                cat_name = cat_info["category"]
                color = cat_info["color"]
                icon = cat_info["icon"]

                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                # Generate small crop
                crop_x1 = max(0, int(x1))
                crop_y1 = max(0, int(y1))
                crop_x2 = min(img_w, int(x2))
                crop_y2 = min(img_h, int(y2))
                crop_b64 = None
                if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                    cropped = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    # Resize to max 120x120 for lightweight transfer
                    cropped.thumbnail((120, 120))
                    crop_b64 = image_to_base64(cropped, format="JPEG")

                det_obj = {
                    "id": idx + 1,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "category": cat_name,
                    "confidence": round(conf * 100, 1),
                    "color": color,
                    "icon": icon,
                    "bbox_pixels": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "bbox_normalized": [round(nx1, 4), round(ny1, 4), round(nw, 4), round(nh, 4)],
                    "width": round(x2 - x1, 1),
                    "height": round(y2 - y1, 1),
                    "crop_base64": crop_b64
                }

                # Pose keypoints if available
                if result.keypoints is not None and idx < len(result.keypoints):
                    kp_data = result.keypoints[idx]
                    kpts = []
                    if kp_data.xy is not None and len(kp_data.xy[0]) > 0:
                        xy_arr = kp_data.xy[0].tolist()
                        conf_arr = kp_data.conf[0].tolist() if kp_data.conf is not None else [1.0] * len(xy_arr)
                        for k_idx, pt in enumerate(xy_arr):
                            k_name = POSE_KEYPOINTS[k_idx] if k_idx < len(POSE_KEYPOINTS) else f"pt_{k_idx}"
                            kpts.append({
                                "name": k_name,
                                "x": round(pt[0], 1),
                                "y": round(pt[1], 1),
                                "conf": round(conf_arr[k_idx] * 100, 1) if k_idx < len(conf_arr) else 0
                            })
                    det_obj["keypoints"] = kpts

                detections.append(det_obj)

        # Classification top results if task is classify
        classification_top = []
        if result.probs is not None:
            top5_indices = result.probs.top5
            top5_confs = result.probs.top5conf.tolist()
            for rank, (c_id, c_conf) in enumerate(zip(top5_indices, top5_confs)):
                c_name = model.names.get(c_id, f"Class {c_id}")
                classification_top.append({
                    "rank": rank + 1,
                    "class_name": c_name,
                    "confidence": round(float(c_conf) * 100, 2)
                })

        return {
            "success": True,
            "task": task,
            "metrics": {
                "total_detections": len(detections),
                "inference_time_ms": inference_time_ms,
                "preprocess_time_ms": preprocess_time_ms,
                "postprocess_time_ms": postprocess_time_ms,
                "total_time_ms": round(total_time, 1),
                "fps": round(1000.0 / max(1.0, inference_time_ms), 1),
                "device_used": target_device,
                "image_width": img_w,
                "image_height": img_h
            },
            "category_counts": category_counts,
            "class_counts": class_counts,
            "detections": detections,
            "classification_top": classification_top,
            "annotated_image": annotated_base64,
            "original_image": original_base64
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

DIST_DIR = os.path.join(BASE_DIR, "static_dist")
if os.path.exists(DIST_DIR):
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# Mount static directory for assets/samples
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    if os.path.exists(DIST_DIR):
        dist_index = os.path.join(DIST_DIR, "index.html")
        if os.path.exists(dist_index):
            return FileResponse(dist_index)
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "YOLOv8x API running. Static index.html not yet created."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
