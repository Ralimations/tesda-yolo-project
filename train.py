import argparse

import torch
from ultralytics import YOLO


def default_device():
    return "0" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train the vehicle detector.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model or checkpoint path.")
    parser.add_argument("--data", default="data.yaml", help="YOLO dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=default_device(), help="Use 'cpu' or a CUDA index such as '0'.")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=args.device,
    )

    print("Training finished successfully!")

if __name__ == "__main__":
    main()
