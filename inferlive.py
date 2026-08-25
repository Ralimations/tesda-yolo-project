import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "testvideos"
RUNS_DIR = BASE_DIR / "runs" / "detect"


def default_device():
    return "0" if torch.cuda.is_available() else "cpu"


def find_default_model():
    candidates = sorted(RUNS_DIR.glob("train*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else BASE_DIR / "yolo11n.pt"


def main():
    parser = argparse.ArgumentParser(description="Run live-style YOLO video inference.")
    parser.add_argument("--model", default=str(find_default_model()))
    parser.add_argument("--source", default=str(VIDEO_DIR / "testvideo.mp4"))
    parser.add_argument("--output", default=str(BASE_DIR / "output_detected.mp4"))
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-window", action="store_true", help="Do not open a preview window.")
    args = parser.parse_args()

    model = YOLO(args.model)
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        frame_count += 1

        if not args.no_window:
            cv2.imshow("YOLO Live Video Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()
    out.release()
    if not args.no_window:
        cv2.destroyAllWindows()

    print(f"Saved {frame_count} frame(s) to {args.output}")


if __name__ == "__main__":
    main()
