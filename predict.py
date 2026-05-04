"""Run YOLOv8 mask-compliance detection on images, video, or webcam.

Draws colour-coded bounding boxes:
  🟢  with_mask            (green)
  🔴  without_mask         (red)
  🟡  mask_weared_incorrect (yellow)

Also overlays a real-time compliance counter on each frame.

Usage::

    python predict.py --image path/to/image.jpg
    python predict.py --input-dir path/to/images/
    python predict.py --video path/to/video.mp4
    python predict.py --webcam
    python predict.py --webcam --weights artifacts/yolo/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = "artifacts/yolo/train/weights/best.pt"

# BGR colours for OpenCV
BOX_COLORS = {
    "with_mask":             (46, 204, 113),   # green
    "without_mask":          (60, 76, 231),    # red
    "mask_weared_incorrect": (15, 196, 241),   # yellow
}
VIOLATION_CLASSES = {"without_mask", "mask_weared_incorrect"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8 real-time mask compliance detection.")
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Path to trained YOLOv8 weights (.pt).",
    )
    parser.add_argument("--image",     default=None, help="Path to a single image.")
    parser.add_argument("--input-dir", default=None, help="Directory of images to annotate.")
    parser.add_argument("--video",     default=None, help="Path to a video file.")
    parser.add_argument("--webcam",    action="store_true", help="Use the default webcam (index 0).")
    parser.add_argument(
        "--output-dir",
        default="predictions",
        help="Directory where annotated outputs will be written.",
    )
    parser.add_argument("--conf",  type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou",   type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int,   default=640,  help="Inference image size.")
    parser.add_argument(
        "--device",
        default="",
        help="Device override ('cpu', '0', 'mps'). Empty = auto.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable the compliance counter overlay.",
    )
    return parser


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(weights_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"Ultralytics not installed: {exc}") from exc

    weights = Path(weights_path)
    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train first:  python train_yolo.py"
        )
    return YOLO(str(weights))


# ---------------------------------------------------------------------------
# Bounding-box drawing helpers
# ---------------------------------------------------------------------------

def draw_boxes(frame: np.ndarray, result, class_names: dict) -> dict[str, int]:
    """Draw YOLO detections on frame; return per-class counts."""
    counts: dict[str, int] = {name: 0 for name in class_names.values()}

    if result.boxes is None or len(result.boxes) == 0:
        return counts

    for box in result.boxes:
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        label    = class_names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = BOX_COLORS.get(label, (255, 255, 255))

        # Rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        text = f"{label.replace('_', ' ')} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bg_y1 = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame, text,
            (x1 + 3, max(th, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )
        counts[label] = counts.get(label, 0) + 1

    return counts


def draw_compliance_overlay(frame: np.ndarray, counts: dict[str, int]) -> None:
    """Draw a semi-transparent compliance panel in the top-right corner."""
    total      = sum(counts.values())
    with_mask  = counts.get("with_mask", 0)
    violations = sum(counts.get(c, 0) for c in VIOLATION_CLASSES)
    pct        = (with_mask / total * 100) if total > 0 else 100.0

    # Panel dims
    ph, pw = frame.shape[:2]
    panel_w, panel_h = 260, 120
    x0, y0 = pw - panel_w - 12, 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Border
    border_color = (60, 76, 231) if violations > 0 else (46, 204, 113)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), border_color, 2)

    lines = [
        ("Compliance Monitor", (200, 200, 200), 0.48, 1),
        (f"With Mask  : {with_mask}", (46, 204, 113), 0.50, 1),
        (f"No Mask    : {counts.get('without_mask', 0)}", (60, 76, 231), 0.50, 1),
        (f"Incorrect  : {counts.get('mask_weared_incorrect', 0)}", (15, 196, 241), 0.50, 1),
        (f"Compliance : {pct:.0f}%", border_color, 0.55, 2),
    ]
    for i, (txt, col, scale, thick) in enumerate(lines):
        cv2.putText(
            frame, txt,
            (x0 + 8, y0 + 18 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSTracker:
    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return len(self._times) / sum(self._times)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def process_frame(frame, model, class_names, conf, iou, imgsz, show_overlay, device):
    results = model.predict(
        frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device if device else None,
        verbose=False,
    )
    counts = draw_boxes(frame, results[0], class_names)
    if show_overlay:
        draw_compliance_overlay(frame, counts)
    return frame, counts


def run_on_capture(capture, model, class_names, args, output_path: Path | None = None):
    """Process a cv2.VideoCapture (webcam or video file)."""
    fps_tracker = FPSTracker()
    writer: cv2.VideoWriter | None = None

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_src = capture.get(cv2.CAP_PROP_FPS) or 25
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_src, (w, h))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame, counts = process_frame(
                frame, model, class_names,
                args.conf, args.iou, args.imgsz,
                not args.no_overlay,
                args.device,
            )

            fps = fps_tracker.tick()
            draw_fps(frame, fps)

            if writer:
                writer.write(frame)

            cv2.imshow("Mask Compliance Detection  [Q to quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    if not any([args.image, args.input_dir, args.video, args.webcam]):
        raise SystemExit("Provide --image, --input-dir, --video, or --webcam.")

    model = load_model(args.weights)
    class_names: dict[int, str] = model.names
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Single image ─────────────────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Cannot read image: {args.image}")
        frame, counts = process_frame(
            frame, model, class_names,
            args.conf, args.iou, args.imgsz,
            not args.no_overlay, args.device,
        )
        out_path = output_dir / Path(args.image).name
        cv2.imwrite(str(out_path), frame)
        print(f"[predict] Saved: {out_path} | Counts: {counts}")

    # ── Image directory ───────────────────────────────────────────────────────
    if args.input_dir:
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for img_path in sorted(Path(args.input_dir).iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"[predict] Skipping: {img_path}")
                continue
            frame, counts = process_frame(
                frame, model, class_names,
                args.conf, args.iou, args.imgsz,
                not args.no_overlay, args.device,
            )
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), frame)
            print(f"[predict] Saved: {out_path} | {counts}")

    # ── Video file ────────────────────────────────────────────────────────────
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {args.video}")
        out_path = output_dir / (Path(args.video).stem + "_annotated.mp4")
        print(f"[predict] Processing video → {out_path}")
        run_on_capture(cap, model, class_names, args, output_path=out_path)
        print(f"[predict] Saved annotated video: {out_path}")

    # ── Webcam ────────────────────────────────────────────────────────────────
    if args.webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise SystemExit("Cannot open webcam.")
        print("[predict] Webcam running. Press Q to quit.")
        run_on_capture(cap, model, class_names, args, output_path=None)


if __name__ == "__main__":
    main()
