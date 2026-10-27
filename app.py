"""
Pothole Detection Web Application
Flask backend for real-time and file-based pothole detection using YOLOv8.
"""

import atexit
import logging
import os
import threading
import time
import uuid

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from utils import draw_box, ensure_dir, get_pothole_color

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "best_pothole.pt" if os.path.exists("best_pothole.pt") else "yolov8n.pt"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

ensure_dir(UPLOAD_FOLDER)
ensure_dir(OUTPUT_FOLDER)

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
_lock = threading.Lock()
detection_active: bool = False
confidence_threshold: float = 0.5
current_fps: float = 0.0
total_detections: int = 0
cap: cv2.VideoCapture | None = None
model: YOLO | None = None

# Job tracker for background processing: job_id -> dict
upload_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Model & Camera Helpers
# ---------------------------------------------------------------------------
def initialize_model(path: str = MODEL_PATH) -> bool:
    """Load YOLOv8 model. Returns True on success."""
    global model
    try:
        model = YOLO(path)
        logger.info("Model loaded from '%s'.", path)
        return True
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return False


def initialize_camera(index: int = 0) -> bool:
    """Open the default webcam. Returns True on success."""
    global cap
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.warning("Could not open camera %d.", index)
        return False
    logger.info("Camera %d opened.", index)
    return True


def release_camera() -> None:
    """Release the webcam resource on shutdown."""
    global cap
    if cap and cap.isOpened():
        cap.release()
        logger.info("Camera released.")


atexit.register(release_camera)


# ---------------------------------------------------------------------------
# Core Detection Logic
# ---------------------------------------------------------------------------
def process_frame(frame, current_model: YOLO, conf: float) -> tuple:
    """
    Run YOLOv8 inference on a single frame.

    Args:
        frame:         BGR image (numpy array).
        current_model: Loaded YOLO model.
        conf:          Confidence threshold.

    Returns:
        (annotated_frame, list[dict]) where each dict has keys
        ``xyxy``, ``conf``, and ``label``.
    """
    detections = []
    results = current_model(frame, conf=conf, verbose=False)

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            score = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = (
                current_model.names[cls_id]
                if hasattr(current_model, "names") and current_model.names
                else f"class_{cls_id}"
            )

            if "pothole" in label.lower():
                detections.append({"xyxy": xyxy, "conf": score, "label": "pothole"})
                frame = draw_box(frame, xyxy, "pothole", score)

    return frame, detections


# ---------------------------------------------------------------------------
# MJPEG Stream Generator
# ---------------------------------------------------------------------------
def gen_frames():
    """Generator that yields MJPEG frames from the webcam."""
    global current_fps, total_detections

    prev_time = time.time()

    while True:
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        with _lock:
            active = detection_active
            conf = confidence_threshold

        if active and model is not None:
            frame, dets = process_frame(frame, model, conf)
            with _lock:
                total_detections += len(dets)

        # FPS calculation
        now = time.time()
        elapsed = now - prev_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        prev_time = now

        # Encode JPEG
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# ---------------------------------------------------------------------------
# File Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _process_image_job(filepath: str, filename: str, job_id: str) -> None:
    """Detect potholes in an image and save the result."""
    try:
        frame = cv2.imread(filepath)
        if frame is None:
            raise ValueError(f"Cannot read image: {filepath}")
        with _lock:
            conf = confidence_threshold
        annotated, dets = process_frame(frame, model, conf)
        out_name = f"processed_{filename}"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        cv2.imwrite(out_path, annotated)
        upload_jobs[job_id] = {
            "status": "completed",
            "image_url": f"/processed/{out_name}",
            "download_url": f"/download/{out_name}",
            "detections": len(dets),
        }
        logger.info("Image job %s complete (%d detections).", job_id, len(dets))
    except Exception as exc:
        logger.error("Image job %s failed: %s", job_id, exc)
        upload_jobs[job_id] = {"status": "error", "message": str(exc)}


def _process_video_job(filepath: str, filename: str, job_id: str) -> None:
    """Detect potholes in a video and save the annotated output."""
    cap_v = cv2.VideoCapture(filepath)
    if not cap_v.isOpened():
        upload_jobs[job_id] = {"status": "error", "message": "Cannot open video."}
        return

    try:
        fps_v = cap_v.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))

        out_name = f"processed_{filename}"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_v, (width, height))

        with _lock:
            conf = confidence_threshold

        frame_count = 0
        det_count = 0
        while cap_v.isOpened():
            ret, frame = cap_v.read()
            if not ret:
                break
            annotated, dets = process_frame(frame, model, conf)
            det_count += len(dets)
            writer.write(annotated)
            frame_count += 1
            progress = int(frame_count / max(total_frames, 1) * 100)
            upload_jobs[job_id]["progress"] = progress

        upload_jobs[job_id] = {
            "status": "completed",
            "filename": out_name,
            "download_url": f"/download/{out_name}",
            "frames_processed": frame_count,
            "detections": det_count,
            "progress": 100,
        }
        logger.info("Video job %s done (%d frames, %d detections).", job_id, frame_count, det_count)
    except Exception as exc:
        logger.error("Video job %s failed: %s", job_id, exc)
        upload_jobs[job_id] = {"status": "error", "message": str(exc)}
    finally:
        cap_v.release()
        writer.release()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start", methods=["POST"])
def start_detection():
    global detection_active
    with _lock:
        detection_active = True
    logger.info("Detection started.")
    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop_detection():
    global detection_active
    with _lock:
        detection_active = False
    logger.info("Detection stopped.")
    return jsonify({"status": "stopped"})


@app.route("/reset_stats", methods=["POST"])
def reset_stats():
    global total_detections
    with _lock:
        total_detections = 0
    return jsonify({"status": "reset"})


@app.route("/set_confidence", methods=["POST"])
def set_confidence():
    global confidence_threshold
    data = request.get_json(silent=True) or {}
    try:
        val = float(data["confidence"])
        if not (0.0 <= val <= 1.0):
            raise ValueError("Out of range")
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    with _lock:
        confidence_threshold = val
    return jsonify({"status": "updated", "confidence": val})


@app.route("/get_stats")
def get_stats():
    with _lock:
        active = detection_active
        conf = confidence_threshold
        fps_val = current_fps
        dets = total_detections
    return jsonify({
        "fps": round(fps_val, 2),
        "confidence": conf,
        "detection_active": active,
        "total_detections": dets,
    })


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed."}), 400

    if model is None:
        return jsonify({"error": "Model not loaded. Cannot process file."}), 503

    filename = secure_filename(file.filename)
    # Prepend UUID to avoid collisions
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    job_id = uuid.uuid4().hex
    upload_jobs[job_id] = {"status": "processing", "progress": 0}

    ext = filename.rsplit(".", 1)[1].lower()
    if ext in {"png", "jpg", "jpeg"}:
        # Images are fast — process synchronously so the UI gets the URL immediately
        _process_image_job(filepath, unique_name, job_id)
        return jsonify({"message": "Image processed.", "job_id": job_id})
    else:
        thread = threading.Thread(
            target=_process_video_job, args=(filepath, unique_name, job_id), daemon=True
        )
        thread.start()
        return jsonify({"message": "Video processing started.", "job_id": job_id})


@app.route("/upload_status/<job_id>")
def upload_status(job_id: str):
    job = upload_jobs.get(job_id, {"status": "not_found"})
    return jsonify(job)


@app.route("/download/<path:filename>")
def download_file(filename: str):
    safe = secure_filename(filename)
    path = os.path.join(OUTPUT_FOLDER, safe)
    if not os.path.exists(path):
        return jsonify({"error": "File not found."}), 404
    return send_file(path, as_attachment=True)


@app.route("/processed/<path:filename>")
def processed_image(filename: str):
    safe = secure_filename(filename)
    path = os.path.join(OUTPUT_FOLDER, safe)
    if not os.path.exists(path):
        return jsonify({"error": "File not found."}), 404
    return send_file(path, mimetype="image/jpeg")


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large (max 500 MB)."}), 413


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model_ok = initialize_model()
    cam_ok = initialize_camera()

    if not model_ok:
        logger.warning("Model failed to load. File processing will be unavailable.")
    if not cam_ok:
        logger.warning("Camera unavailable. Live feed will be inactive.")

    # Run with threaded=True for concurrent MJPEG stream + API calls
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
