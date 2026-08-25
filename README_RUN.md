# TESDA YOLO Project

This project contains a YOLO object-detection dataset and Flask control panel for two classes:

- `vehicle_body`
- `vehicle_plate`

## Conda Setup

```powershell
conda env create -f environment.yml
conda activate tesda-yolo
```

For an NVIDIA GPU setup, remove `cpuonly` from `environment.yml` and install the PyTorch CUDA build that matches your driver.

## Smoke Tests

```powershell
python train.py --epochs 1 --batch 2 --imgsz 320 --device cpu
python inference.py --device cpu --max-frames 5
python app.py
```

Open the Flask app at:

```text
http://127.0.0.1:5000
```

The archive did not include trained `best.pt` weights under `runs/detect/*/weights/`, so inference falls back to `yolo11n.pt` until a new training run creates custom weights.
