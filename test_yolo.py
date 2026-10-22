#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlretrieve

from yolo_edge import export_model, parse_srt_boxes, simulate_box_data_for_frame


def generate_sample_image(destination: str | Path, width: int = 1280, height: int = 720) -> Path:
    """Create a local fallback image that simulates an industrial inspection scene."""
    import cv2
    import numpy as np

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    image = np.full((height, width, 3), 35, dtype=np.uint8)
    image[:, :, 1] = 90
    cv2.rectangle(image, (160, 180), (520, 520), (18, 110, 245), thickness=3)
    cv2.rectangle(image, (720, 200), (980, 480), (18, 200, 160), thickness=3)
    cv2.circle(image, (1040, 150), 80, (0, 200, 255), thickness=4)
    cv2.putText(image, "industrial inspection", (120, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite(str(target), image)
    return target


def generate_sample_video(destination: str | Path, width: int = 640, height: int = 360, fps: int = 12, seconds: int = 4) -> Path:
    """Generate a tiny synthetic drone-like clip when no sample video is available."""
    import cv2
    import numpy as np

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create sample video at {target}")

    for frame_index in range(fps * seconds):
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        frame[:, :, 1] = 90
        offset = int((frame_index / max(1, fps * seconds)) * (width - 240))
        cv2.rectangle(frame, (120 + offset, 70), (260 + offset, 220), (18, 110, 245), 3)
        cv2.rectangle(frame, (400 - offset, 120), (560 - offset, 260), (18, 200, 160), 3)
        cv2.circle(frame, (520, 80 + (frame_index % 60)), 40, (0, 200, 255), 4)
        cv2.putText(frame, "drone test", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    return target


def download_sample_asset(url: str, destination: str | Path) -> Path:
    """Download a sample input asset for local inference when network access is available."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    video_suffixes = {".mp4", ".avi", ".mov", ".mkv"}
    image_suffixes = {".jpg", ".jpeg", ".png"}

    if target.suffix.lower() in image_suffixes:
        try:
            urlretrieve(url, str(target))
            return target
        except URLError:
            return generate_sample_image(target)

    if target.suffix.lower() in video_suffixes:
        try:
            urlretrieve(url, str(target))
            return target
        except URLError:
            return generate_sample_video(target)

    try:
        urlretrieve(url, str(target))
        return target
    except URLError:
        if Path(url).suffix.lower() in video_suffixes or target.suffix.lower() in video_suffixes:
            return generate_sample_video(target.with_suffix(".mp4"))
        fallback = generate_sample_image(target.with_suffix(".jpg"))
        return fallback


def build_demo_video(
    source: Path,
    srt_path: str | None = None,
    srt_boxes: list | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create a synthetic annotated demo from a real source using SRT box data."""
    import cv2

    source_path = Path(source)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source video: {source_path}")

    srt_boxes = srt_boxes if srt_boxes is not None else (parse_srt_boxes(srt_path) if srt_path else [])
    output = Path(output_path) if output_path else source_path.with_name(source_path.stem + "_synthetic.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if srt_boxes:
            boxes = simulate_box_data_for_frame((height, width, 3), srt_boxes, frame_index / fps, frame_index)
            for entry in boxes:
                x1, y1, x2, y2 = [int(v) for v in entry["xyxy"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, entry["label"], (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    return output


def prepare_source(source: str) -> str:
    """Return a local source path; download remote assets when possible or fallback to synthetic local data."""
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"}:
        return source

    sample_path = Path("samples") / Path(parsed.path).name
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = download_sample_asset(source, sample_path)
    except Exception:
        if sample_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            downloaded = generate_sample_video(sample_path)
        else:
            downloaded = generate_sample_image(sample_path.with_suffix(".jpg"))
    return str(downloaded)


def resolve_srt_input(source: str, srt_path: str | None) -> tuple[str, str | None]:
    """Allow SRT files to be passed either via --srt or as the main source file."""
    source_path = Path(source)
    if srt_path is None and source_path.suffix.lower() == ".srt":
        fallback_image = generate_sample_image(Path("samples") / "synthetic_scene.jpg")
        return str(fallback_image), str(source_path)
    return source, srt_path


def run_inference(
    model_name: str,
    source: str,
    conf: float,
    device: str,
    srt_path: str | None = None,
    export_format: str | None = None,
    save_output: bool = False,
    classes: list[int] | None = None,
) -> None:
    """Run a YOLO11 inference pass on a local or remote image/video source."""
    source, srt_path = resolve_srt_input(source, srt_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Install dependencies with: pip install -r requirements.txt") from exc

    model = YOLO(model_name)

    if export_format:
        export_model(model, export_format=export_format)
        print(f"Model exported to edge format '{export_format}'.")
        return

    local_source = prepare_source(source)
    source_path = Path(local_source)

    if srt_path:
        srt_boxes = parse_srt_boxes(srt_path)
        if source_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            build_demo_video(source_path, srt_boxes=srt_boxes, output_path=Path("demo_output.mp4"))
            print(f"Synthetic SRT demo saved to demo_output.mp4 using {len(srt_boxes)} boxes.")
            return
        if srt_boxes:
            print(f"Loaded {len(srt_boxes)} synthetic SRT boxes for image simulation.")

    output = Path("runs/detect")
    output.mkdir(parents=True, exist_ok=True)
    is_video = source_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}

    if is_video:
        model(local_source, conf=conf, device=device, save=save_output, classes=classes)
        if save_output:
            annotated_path = output / "annotated_video.mp4"
            if annotated_path.exists():
                print(f"Video inference complete! Annotated results saved to {annotated_path}.")
            else:
                print(f"Video inference completed; output saved to runs/detect.")
        else:
            print("Video inference complete without saving an output file.")
        return

    results = model(local_source, conf=conf, device=device, classes=classes)
    if not results:
        print(f"No detections produced for {local_source}.")
        return

    results[0].save(filename=str(output / "result.jpg"))
    print(f"Detection complete! Check {output / 'result.jpg'}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11 inference + edge export smoke-test")
    parser.add_argument("--model", default="yolo11n.pt", help="Model checkpoint to load, e.g. yolo11n.pt")
    parser.add_argument("--source", default="https://ultralytics.com/images/bus.jpg", help="Image or video file path or URL")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu, 0, or mps)")
    parser.add_argument("--srt", help="Optional path to an SRT file containing synthetic box rectangles")
    parser.add_argument("--export", choices=["onnx", "tflite", "litert"], help="Export the model to an edge-ready format")
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save the rendered output video to disk",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="Filter detections by class IDs (e.g., 2 for car, 5 for bus, 7 for truck)",
    )
    args = parser.parse_args()

    run_inference(
        model_name=args.model,
        source=args.source,
        conf=args.conf,
        device=args.device,
        srt_path=args.srt,
        export_format=args.export,
        save_output=args.save_output,
        classes=args.classes,
    )


if __name__ == "__main__":
    main()
