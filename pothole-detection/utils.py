"""
Utility functions for the Pothole Detection project.
"""

import csv
import logging
import os
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR color for pothole bounding boxes (bright cyan-yellow)
_POTHOLE_COLOR: Tuple[int, int, int] = (0, 220, 255)


def get_pothole_color() -> Tuple[int, int, int]:
    """Return the BGR color used to draw pothole bounding boxes."""
    return _POTHOLE_COLOR


def draw_box(
    frame: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    label: str,
    conf: float,
) -> np.ndarray:
    """
    Draw a bounding box with label and confidence score on *frame*.

    Args:
        frame: BGR image array (modified in-place).
        xyxy:  Bounding box as (x1, y1, x2, y2).
        label: Class name string.
        conf:  Confidence score in [0, 1].

    Returns:
        The annotated frame.
    """
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    color = get_pothole_color()

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

    # Build label text
    text = f"{label}: {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Filled background rectangle so the label is always readable
    bg_y1 = max(y1 - text_h - baseline - 4, 0)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + text_w + 4, y1), color, cv2.FILLED)
    cv2.putText(
        frame,
        text,
        (x1 + 2, y1 - baseline - 2),
        font,
        font_scale,
        (0, 0, 0),  # Black text on colored background
        thickness,
        cv2.LINE_AA,
    )
    return frame


def ensure_dir(path: str) -> None:
    """Create *path* and any missing parent directories."""
    os.makedirs(path, exist_ok=True)


def save_frame(frame: np.ndarray, idx: int, out_dir: str) -> str:
    """
    Save a video frame as a JPEG image.

    Args:
        frame:   BGR frame array.
        idx:     Zero-based frame index (used in filename).
        out_dir: Directory to write the file into.

    Returns:
        Full path of the saved file.
    """
    ensure_dir(out_dir)
    filename = f"frame_{idx:06d}.jpg"
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def log_detection(
    timestamp: str,
    frame_ref: object,
    label: str,
    conf: float,
    xyxy: Tuple[int, int, int, int],
    csv_path: str,
) -> None:
    """
    Append one detection row to a CSV log file.

    The file is created with a header on the first write; subsequent calls
    append rows without re-writing the header.

    Args:
        timestamp:  ISO-8601 string (e.g. from ``datetime.now().isoformat()``).
        frame_ref:  Frame index (int) or image filename (str).
        label:      Detected class name.
        conf:       Confidence score.
        xyxy:       Bounding box (x1, y1, x2, y2).
        csv_path:   Destination CSV file path.
    """
    ensure_dir(os.path.dirname(os.path.abspath(csv_path)))

    x1, y1, x2, y2 = (int(v) for v in xyxy)
    row = {
        "timestamp": timestamp,
        "frame_ref": frame_ref,
        "label": label,
        "confidence": round(float(conf), 4),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "area_px": (x2 - x1) * (y2 - y1),
    }

    fieldnames = list(row.keys())
    write_header = not os.path.exists(csv_path)

    try:
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError as exc:
        logger.error("Failed to write detection log: %s", exc)


def compute_severity(area_px: int, frame_area: int) -> str:
    """
    Classify a pothole as Low / Medium / High based on the fraction
    of the frame it occupies.

    Args:
        area_px:    Bounding-box pixel area.
        frame_area: Total frame pixel area (width * height).

    Returns:
        One of ``"Low"``, ``"Medium"``, or ``"High"``.
    """
    if frame_area == 0:
        return "Unknown"
    ratio = area_px / frame_area
    if ratio < 0.01:
        return "Low"
    if ratio < 0.05:
        return "Medium"
    return "High"
