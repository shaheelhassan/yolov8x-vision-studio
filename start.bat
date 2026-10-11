@echo off
title YOLOv8x AI Vision Studio
echo ===================================================
echo Starting YOLOv8x AI Vision Studio (FastAPI)
echo Model: yolov8x.pt (68M Parameters)
echo URL:   http://127.0.0.1:8000
echo ===================================================
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
