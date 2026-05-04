"""Evaluate a trained YOLOv8 model for mask compliance detection.

Reports:
  • mAP@0.5 and mAP@0.5:0.95 (overall and per-class)
  • Precision and Recall (overall and per-class)
  • Inference FPS
  • Confusion matrix plot
  • Precision-Recall curves

Outputs are saved to artifacts/evaluation/ by default.

Usage::

    python evaluate.py
    python evaluate.py --weights artifacts/yolo/train/weights/best.pt
    python evaluate.py --split test --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model — mAP, Precision, Recall, FPS.")
    parser.add_argument(
        "--weights",
        default="artifacts/yolo/train/weights/best.pt",
        help="Path to trained YOLOv8 weights (.pt file).",
    )
    parser.add_argument(
        "--yaml",
        default="dataset.yaml",
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou",  type=float, default=0.6, help="NMS IoU threshold.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/evaluation",
        help="Directory to save evaluation results and plots.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device override ('cpu', '0', 'mps'). Empty = auto.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(
            f"Weights not found: {weights_path}\n"
            "Train first:  python train_yolo.py"
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"Ultralytics not installed: {exc}") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[evaluate] Loading weights: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[evaluate] Running validation on split='{args.split}' …\n")
    metrics = model.val(
        data=args.yaml,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
        project=str(output_dir),
        name="val_run",
        exist_ok=True,
        plots=True,        # saves PR curve, confusion matrix, etc.
        save_json=True,
    )

    # ── Extract key numbers ──────────────────────────────────────────────────
    map50     = float(metrics.box.map50)
    map50_95  = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall    = float(metrics.box.mr)
    fps       = 1000.0 / metrics.speed["inference"] if metrics.speed["inference"] > 0 else 0.0

    class_names = model.names  # {0: 'with_mask', 1: 'without_mask', ...}
    per_class: list[dict] = []
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            per_class.append({
                "class": class_names.get(int(cls_idx), str(cls_idx)),
                "precision": float(metrics.box.p[i]),
                "recall":    float(metrics.box.r[i]),
                "ap50":      float(metrics.box.ap50[i]),
                "ap50_95":   float(metrics.box.ap[i]),
            })

    results_dict = {
        "weights": str(weights_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "overall": {
            "mAP@0.5":      map50,
            "mAP@0.5:0.95": map50_95,
            "precision":    precision,
            "recall":       recall,
            "fps":          round(fps, 1),
        },
        "per_class": per_class,
    }

    results_path = output_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")

    # ── Copy Ultralytics-generated plots to output_dir ───────────────────────
    val_run_dir = output_dir / "val_run"
    if val_run_dir.exists():
        for plot_file in val_run_dir.glob("*.png"):
            shutil.copy2(plot_file, output_dir / plot_file.name)

    # ── Print human-readable summary ─────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  EVALUATION RESULTS — Mask Compliance Monitor")
    print("=" * 56)
    print(f"  mAP@0.5       : {map50:.4f}  ({map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95  : {map50_95:.4f}  ({map50_95*100:.2f}%)")
    print(f"  Precision      : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall         : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  FPS (inference): {fps:.1f}")
    print("-" * 56)
    if per_class:
        print(f"  {'Class':<28} {'P':>6}  {'R':>6}  {'AP50':>7}")
        print("-" * 56)
        for pc in per_class:
            print(
                f"  {pc['class']:<28} "
                f"{pc['precision']:>6.3f}  "
                f"{pc['recall']:>6.3f}  "
                f"{pc['ap50']:>7.4f}"
            )
    print("=" * 56)
    print(f"\n  Results JSON : {results_path}")
    print(f"  Plots dir    : {output_dir}\n")
    print("Next step — run live detection:")
    print(f"  python predict.py --webcam --weights {weights_path}")
    print(f"  python compliance_server.py --weights {weights_path}\n")


if __name__ == "__main__":
    main()
