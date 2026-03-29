"""
CLI tool for pothole detection on webcam, video files, or image folders.

Usage examples
--------------
Webcam (default):
    python detect_realtime.py

Video file with GPU, saving output and logging:
    python detect_realtime.py --source road.mp4 --model best_pothole.pt \\
        --device cuda --save --log --conf 0.3

Image folder:
    python detect_realtime.py --source images/ --save --log

Press  q  to quit the webcam / video window.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from utils import draw_box, ensure_dir, log_detection, save_frame, compute_severity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def process_frame(frame, model: YOLO, conf: float) -> tuple:
    """
    Run YOLOv8 inference on a single frame.

    Returns
    -------
    annotated_frame : np.ndarray
    detections      : list[dict]  keys: xyxy, conf, label
    """
    detections = []
    results = model(frame, conf=conf, verbose=False)

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            score = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = (
                model.names[cls_id]
                if hasattr(model, "names") and model.names
                else f"class_{cls_id}"
            )
            if "pothole" in label.lower() or cls_id == 0:
                detections.append({"xyxy": xyxy, "conf": score, "label": "pothole"})
                frame = draw_box(frame, xyxy, "pothole", score)

    return frame, detections


# ---------------------------------------------------------------------------
# Video / Webcam processing
# ---------------------------------------------------------------------------
def process_video_stream(args, model: YOLO, cap: cv2.VideoCapture) -> None:
    """Process a live webcam feed or a video file."""
    out_writer = None
    frames_dir = None

    is_file = not str(args.source).isdigit()
    if args.save and is_file:
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ensure_dir("outputs")
        out_path = "outputs/annotated.mp4"
        out_writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (w, h)
        )
        logger.info("Saving output video to %s", out_path)

    if args.save_frames:
        frames_dir = "outputs/frames/"
        ensure_dir(frames_dir)

    csv_path = "outputs/detections.csv" if args.log else None
    if csv_path:
        ensure_dir("outputs")

    frame_idx = 0
    prev_time = time.time()
    total_dets = 0

    logger.info("Starting video stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, dets = process_frame(frame, model, args.conf)
        total_dets += len(dets)

        # FPS overlay
        if args.show_fps:
            now = time.time()
            fps_val = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                annotated,
                f"FPS: {fps_val:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Detection counter overlay
        h_frame, w_frame = annotated.shape[:2]
        cv2.putText(
            annotated,
            f"Detections: {len(dets)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        if frames_dir:
            save_frame(annotated, frame_idx, frames_dir)

        if out_writer:
            out_writer.write(annotated)

        if csv_path:
            ts = datetime.now().isoformat()
            for det in dets:
                area = int((det["xyxy"][2] - det["xyxy"][0]) * (det["xyxy"][3] - det["xyxy"][1]))
                severity = compute_severity(area, w_frame * h_frame)
                log_detection(ts, frame_idx, f"{det['label']}({severity})", det["conf"], det["xyxy"], csv_path)

        cv2.imshow("Pothole Detection  [q to quit]", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()
    logger.info("Done. Processed %d frames, %d total detections.", frame_idx, total_dets)


# ---------------------------------------------------------------------------
# Image folder processing
# ---------------------------------------------------------------------------
def process_image_folder(args, model: YOLO) -> None:
    """Run detection on every image inside a directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_files = [
        f for f in os.listdir(args.source) if os.path.splitext(f)[1].lower() in exts
    ]
    if not image_files:
        logger.error("No supported images found in '%s'.", args.source)
        return

    out_dir = "outputs/processed_images/"
    ensure_dir(out_dir)
    csv_path = "outputs/detections.csv" if args.log else None
    if csv_path:
        ensure_dir("outputs")

    total_dets = 0
    logger.info("Processing %d images from '%s'...", len(image_files), args.source)

    for filename in tqdm(image_files, unit="img"):
        filepath = os.path.join(args.source, filename)
        frame = cv2.imread(filepath)
        if frame is None:
            logger.warning("Cannot read '%s', skipping.", filepath)
            continue

        annotated, dets = process_frame(frame, model, args.conf)
        total_dets += len(dets)

        if args.save or args.save_frames:
            out_path = os.path.join(out_dir, f"processed_{filename}")
            cv2.imwrite(out_path, annotated)

        if csv_path and dets:
            h_f, w_f = annotated.shape[:2]
            ts = datetime.now().isoformat()
            for det in dets:
                area = int((det["xyxy"][2] - det["xyxy"][0]) * (det["xyxy"][3] - det["xyxy"][1]))
                severity = compute_severity(area, w_f * h_f)
                log_detection(ts, filename, f"{det['label']}({severity})", det["conf"], det["xyxy"], csv_path)

    logger.info("Image processing complete. %d total detections.", total_dets)
    logger.info("Results saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time pothole detection using YOLOv8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index (e.g. 0), video file path, or image folder.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to a YOLOv8 .pt model file.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Detection confidence threshold [0.0–1.0].",
    )
    p.add_argument("--save", action="store_true", help="Save annotated output.")
    p.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual frames to outputs/frames/.",
    )
    p.add_argument(
        "--log",
        action="store_true",
        help="Write detection log to outputs/detections.csv.",
    )
    p.add_argument(
        "--show-fps",
        action="store_true",
        help="Overlay FPS counter on the video output.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Load model
    try:
        model = YOLO(args.model)
        model.to(args.device)
        logger.info("Model '%s' loaded on device '%s'.", args.model, args.device)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        sys.exit(1)

    # Dispatch
    if os.path.isdir(args.source):
        process_image_folder(args, model)
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error("Cannot open source: %s", args.source)
            sys.exit(1)
        process_video_stream(args, model, cap)


if __name__ == "__main__":
    main()
