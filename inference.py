import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs" / "detect"
VIDEO_DIR = BASE_DIR / "testvideos"


def default_device():
    return "0" if torch.cuda.is_available() else "cpu"


def find_default_model():
    candidates = sorted(RUNS_DIR.glob("train*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else BASE_DIR / "yolo11n.pt"


def find_default_source():
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 videos found in {VIDEO_DIR}")
    return videos[0]


def run_inference():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a video.")
    parser.add_argument("--model", default=str(find_default_model()), help="Model checkpoint path.")
    parser.add_argument("--source", default=str(find_default_source()), help="Video path.")
    parser.add_argument("--output", default=str(BASE_DIR / "output_detected.mp4"), help="Annotated output video path.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=default_device(), help="Use 'cpu' or a CUDA index such as '0'.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. Use 0 for the full video.")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        result = model(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        writer.write(result.plot())
        frame_count += 1

        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()
    writer.release()
    print(f"Inference completed: {frame_count} frame(s) written to {args.output}")


if __name__ == "__main__":
    run_inference()
