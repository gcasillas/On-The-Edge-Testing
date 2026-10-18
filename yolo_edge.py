from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class SRTBox:
    start_seconds: float
    end_seconds: float
    x1: float
    y1: float
    x2: float
    y2: float


def _parse_timestamp(timestamp: str) -> float:
    """Parse SRT time values like '00:01:02,500' into seconds."""
    if " --> " in timestamp:
        start, _ = (part.strip() for part in timestamp.split(" --> ", 1))
        timestamp = start

    if "," in timestamp:
        timestamp, millis = timestamp.split(",", 1)
    else:
        millis = "0"

    hours, minutes, seconds = (float(part) for part in timestamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds + float(millis) / 1000.0


def parse_srt_boxes(srt_path: str | Path) -> List[SRTBox]:
    """Parse an SRT file containing per-frame/segment boxes.

    Accepts payloads such as:
      box: 100, 50, 250, 180
      100,50,250,180
      0.15,0.20,0.45,0.60
    """
    srt_file = Path(srt_path)
    text = srt_file.read_text(encoding="utf-8")

    blocks = re.split(r"\n\s*\n", text.strip())
    boxes: List[SRTBox] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue

        time_line = lines[1]
        if " --> " not in time_line:
            continue

        start_text, end_text = [part.strip() for part in time_line.split(" --> ", 1)]
        start_seconds = _parse_timestamp(start_text)
        end_seconds = _parse_timestamp(end_text)

        payload = " ".join(lines[2:])
        numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", payload)]
        if len(numbers) < 4:
            continue

        x1, y1, x2, y2 = numbers[:4]
        if x1 <= 1 and y1 <= 1 and x2 <= 1 and y2 <= 1:
            x1, y1, x2, y2 = x1 * 1920, y1 * 1080, x2 * 1920, y2 * 1080

        boxes.append(
            SRTBox(
                start_seconds=start_seconds,
                end_seconds=max(end_seconds, start_seconds + 0.1),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
        )

    return boxes


def simulate_box_data_for_frame(
    frame_shape: Sequence[int],
    srt_boxes: Iterable[SRTBox],
    timestamp_seconds: float,
    frame_index: int = 0,
) -> List[dict]:
    """Create synthetic bounding boxes for a frame based on an SRT timeline."""
    height, width = frame_shape[:2]
    simulated: List[dict] = []

    for box in srt_boxes:
        if not (box.start_seconds <= timestamp_seconds < box.end_seconds):
            continue

        jitter_x = math.sin(frame_index * 0.45 + box.start_seconds) * 8.0
        jitter_y = math.cos(frame_index * 0.32 + box.end_seconds) * 8.0
        x1 = max(0.0, min(float(box.x1) + jitter_x, float(width - 1)))
        y1 = max(0.0, min(float(box.y1) + jitter_y, float(height - 1)))
        x2 = max(x1 + 5.0, min(float(box.x2) + jitter_x, float(width - 1)))
        y2 = max(y1 + 5.0, min(float(box.y2) + jitter_y, float(height - 1)))

        simulated.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "confidence": 0.55,
                "class_id": 0,
                "label": "synthetic-object",
            }
        )

    return simulated


def export_model(model, export_format: str, imgsz: int = 640, project: str = "runs/export") -> str:
    """Export a YOLO model to an edge-friendly format."""
    format_name = export_format.lower()
    if format_name not in {"onnx", "tflite"}:
        raise ValueError("export_format must be 'onnx' or 'tflite'")

    export_path = model.export(format=format_name, imgsz=imgsz, project=project)
    return str(export_path)
