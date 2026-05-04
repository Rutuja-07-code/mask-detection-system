"""Flask compliance server — streams webcam MJPEG and exposes live JSON stats.

Serves the HTML dashboard at http://localhost:5050/

Endpoints:
  GET /                    → dashboard HTML
  GET /video_feed          → MJPEG stream
  GET /api/stats           → JSON: live counts + compliance %
  GET /api/history         → JSON: last-N-seconds timeline
  POST /api/reset          → reset session counters

Usage::

    python compliance_server.py
    python compliance_server.py --weights artifacts/yolo/train/weights/best.pt --port 5050
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = "artifacts/yolo/train/weights/best.pt"
TRAINING_SUMMARY = Path("artifacts/yolo/training_summary.json")
VIOLATION_CLASSES = {"without_mask", "mask_weared_incorrect"}
BOX_COLORS = {
    "with_mask":             (46, 204, 113),
    "without_mask":          (60, 76, 231),
    "mask_weared_incorrect": (15, 196, 241),
}

# ---------------------------------------------------------------------------
# Shared state (thread-safe via lock)
# ---------------------------------------------------------------------------

_lock         = threading.Lock()
_latest_frame: bytes | None = None
_counts: dict[str, int] = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
_history: deque = deque(maxlen=120)   # last 120 ticks (~60 s at 2 Hz)
_session_max_violation_pct: float = 0.0

# ---------------------------------------------------------------------------
# YOLO inference thread
# ---------------------------------------------------------------------------


def _run_detector(weights: str, conf: float, iou: float, imgsz: int, device: str) -> None:
    """Background thread: captures webcam frames, runs YOLO, updates shared state."""
    global _latest_frame, _counts, _session_max_violation_pct

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[server] Ultralytics not installed: {exc}")
        return

    model = YOLO(weights)
    class_names: dict[int, str] = model.names

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[server] WARNING: Cannot open webcam — using blank frames.")
        # Serve blank green frames so the dashboard still loads
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No webcam found", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 76, 231), 2)
        _, jpg = cv2.imencode(".jpg", blank)
        with _lock:
            _latest_frame = jpg.tobytes()
        return

    last_history_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        results = model.predict(
            frame, conf=conf, iou=iou, imgsz=imgsz,
            device=device if device else None, verbose=False,
        )
        result = results[0]

        local_counts: dict[str, int] = {name: 0 for name in class_names.values()}

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = class_names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = BOX_COLORS.get(label, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label.replace('_', ' ')} {conf_score:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                bg_y1 = max(0, y1 - th - 8)
                cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, text, (x1 + 3, max(th, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                local_counts[label] = local_counts.get(label, 0) + 1

        total = sum(local_counts.values())
        violations = sum(local_counts.get(c, 0) for c in VIOLATION_CLASSES)
        viol_pct = (violations / total * 100) if total > 0 else 0.0
        comp_pct = 100.0 - viol_pct

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_bytes = jpg.tobytes()

        now = time.time()
        with _lock:
            _latest_frame = jpg_bytes
            _counts = dict(local_counts)
            if viol_pct > _session_max_violation_pct:
                _session_max_violation_pct = viol_pct
            if now - last_history_ts >= 0.5:   # record every 0.5 s
                _history.append({
                    "ts": round(now, 1),
                    "with_mask": local_counts.get("with_mask", 0),
                    "without_mask": local_counts.get("without_mask", 0),
                    "mask_weared_incorrect": local_counts.get("mask_weared_incorrect", 0),
                    "compliance_pct": round(comp_pct, 1),
                })
                last_history_ts = now

    cap.release()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

DASHBOARD_HTML = (Path(__file__).parent / "dashboard" / "index.html").read_text(encoding="utf-8") \
    if (Path(__file__).parent / "dashboard" / "index.html").exists() else \
    "<h1>Dashboard not found. Run from project root.</h1>"


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


def _mjpeg_generator():
    while True:
        with _lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(1 / 30)  # cap at 30 fps


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def api_stats():
    with _lock:
        counts = dict(_counts)
        max_viol = _session_max_violation_pct

    total      = sum(counts.values())
    violations = sum(counts.get(c, 0) for c in VIOLATION_CLASSES)
    viol_pct   = (violations / total * 100) if total > 0 else 0.0
    comp_pct   = 100.0 - viol_pct

    return jsonify({
        "counts": counts,
        "total": total,
        "violations": violations,
        "compliance_pct": round(comp_pct, 1),
        "violation_pct": round(viol_pct, 1),
        "session_max_violation_pct": round(max_viol, 1),
        "alert": viol_pct > 30.0,
        "timestamp": time.time(),
    })


@app.route("/api/history")
def api_history():
    with _lock:
        history = list(_history)
    return jsonify(history)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global _session_max_violation_pct
    with _lock:
        _history.clear()
        _session_max_violation_pct = 0.0
    return jsonify({"status": "reset"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mask compliance webcam server.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLOv8 weights (.pt).")
    parser.add_argument("--port",    type=int, default=5050, help="Flask port.")
    parser.add_argument("--host",    default="0.0.0.0", help="Flask host.")
    parser.add_argument("--conf",    type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou",     type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz",   type=int,   default=640, help="Inference image size.")
    parser.add_argument("--device",  default="", help="Device ('cpu','0','mps'). Empty=auto.")
    return parser


def resolve_weights(weights_arg: str) -> Path:
    weights = Path(weights_arg)
    if weights.exists():
        return weights

    if weights_arg == DEFAULT_WEIGHTS and TRAINING_SUMMARY.exists():
        try:
            summary = json.loads(TRAINING_SUMMARY.read_text(encoding="utf-8"))
            best_weights = summary.get("best_weights")
            if best_weights:
                candidate = Path(best_weights)
                if candidate.exists():
                    return candidate
        except json.JSONDecodeError:
            pass

    return weights


def main() -> None:
    args = build_parser().parse_args()

    weights = resolve_weights(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train first:  python train_yolo.py\n"
            "Then evaluate: python evaluate.py\n"
            "Or pass --weights with the correct path (see artifacts/yolo/training_summary.json)."
        )

    print(f"\n[server] Starting mask compliance server")
    print(f"  Weights    : {weights}")
    print(f"  Dashboard  : http://localhost:{args.port}/")
    print(f"  Video feed : http://localhost:{args.port}/video_feed")
    print(f"  Stats API  : http://localhost:{args.port}/api/stats\n")

    # Launch detector in background thread
    detector_thread = threading.Thread(
        target=_run_detector,
        args=(str(weights), args.conf, args.iou, args.imgsz, args.device),
        daemon=True,
    )
    detector_thread.start()

    # Brief pause so the first frame is ready
    time.sleep(1.5)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
