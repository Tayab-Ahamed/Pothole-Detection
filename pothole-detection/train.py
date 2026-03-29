"""
Training script for the YOLOv8 pothole detection model.

Quickstart
----------
1. Download a pothole dataset (e.g. from Roboflow) in YOLO format.
2. Place the data.yaml in the project root (see template below).
3. Run:
       python train.py --epochs 100 --model yolov8s.pt --batch 16

The best weights are saved at:
    runs/detect/pothole_detector<N>/weights/best.pt

Copy them to the project root to use with app.py:
    cp runs/detect/pothole_detector/weights/best.pt best_pothole.pt
"""

import argparse
import logging
import os
import sys

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_YAML_TEMPLATE = """\
# ---- data.yaml template ----
# Adjust the paths below to match your dataset layout.
path: .           # project root (all other paths are relative to this)
train: data/train/images
val:   data/val/images
# test: data/test/images   # optional

nc: 1             # number of classes
names: ['pothole']
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a YOLOv8 model for pothole detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to the dataset YAML file.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 base model to fine-tune.",
    )
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size (pixels).")
    p.add_argument("--batch", type=int, default=16, help="Batch size (-1 = auto).")
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="Training device: '' (auto), 'cpu', '0' (GPU 0), '0,1' (multi-GPU).",
    )
    p.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory to save training runs.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="pothole_detector",
        help="Sub-directory name inside --project.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # --- Validate data.yaml ------------------------------------------------
    if not os.path.exists(args.data):
        logger.error("'%s' not found.", args.data)
        logger.info("Create a data.yaml file with the following structure:\n%s", DATA_YAML_TEMPLATE)
        sys.exit(1)

    # --- Load model --------------------------------------------------------
    try:
        model = YOLO(args.model)
        logger.info("Base model '%s' loaded.", args.model)
    except Exception as exc:
        logger.error("Could not load model: %s", exc)
        sys.exit(1)

    # --- Train -------------------------------------------------------------
    logger.info(
        "Starting training: epochs=%d, imgsz=%d, batch=%d, device='%s'",
        args.epochs,
        args.imgsz,
        args.batch,
        args.device or "auto",
    )

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device if args.device else None,
            project=args.project,
            name=args.name,
            resume=args.resume,
            # Recommended augmentation settings for road-surface data
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            flipud=0.0,   # Road images rarely flip vertically
            fliplr=0.5,
            mosaic=1.0,
        )
        best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
        logger.info("Training complete.")
        logger.info("Best weights saved at: %s", best_weights)
        logger.info("To use with app.py, run:  cp %s best_pothole.pt", best_weights)
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
