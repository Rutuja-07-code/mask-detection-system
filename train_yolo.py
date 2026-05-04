"""Train a YOLOv8 object detection model for mask compliance monitoring.

Steps performed automatically:
  1. Convert VOC XML annotations → YOLO TXT format (skipped if data/yolo exists).
  2. Write dataset.yaml.
  3. Train with Ultralytics YOLOv8.
  4. Save best weights + training summary JSON.

Usage::

    python train_yolo.py
    python train_yolo.py --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16
    python train_yolo.py --force-prepare   # re-build the YOLO dataset first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.mask_detection.constants import DEFAULT_RAW_DATASET_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for mask compliance detection.")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_RAW_DATASET_ROOT,
        help="Raw dataset root with 'images' and 'annotations'.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/yolo",
        help="Directory for the prepared YOLO dataset.",
    )
    parser.add_argument(
        "--yaml-path",
        default="dataset.yaml",
        help="Path to the dataset YAML file (written during preparation).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 backbone to fine-tune (nano is fastest; large is most accurate).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 = auto).")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/yolo",
        help="Directory for training runs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Re-build the YOLO dataset even if it already exists.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device override, e.g. 'cpu', '0', 'mps'. Empty = auto-detect.",
    )
    return parser


def ensure_yolo_dataset(args) -> str:
    """Prepare the YOLO dataset if needed and return the yaml path."""
    from src.mask_detection.dataset_tools import prepare_yolo_dataset

    data_dir = Path(args.data_dir)
    train_images = data_dir / "train" / "images"

    if args.force_prepare or not train_images.exists():
        print("[train_yolo] Preparing YOLO dataset …")
        prepare_yolo_dataset(
            dataset_root=args.dataset_root,
            output_dir=data_dir,
            yaml_path=args.yaml_path,
            force=args.force_prepare,
        )
    else:
        print(f"[train_yolo] Using existing YOLO dataset at: {data_dir}")

    return args.yaml_path


def main() -> None:
    args = build_parser().parse_args()

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    yaml_path = ensure_yolo_dataset(args)

    # ── 2. Import Ultralytics ────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run:  pip install ultralytics>=8.2\n"
            f"Original error: {exc}"
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. Train ────────────────────────────────────────────────────────────
    print(f"\n[train_yolo] Starting YOLOv8 training")
    print(f"  Model        : {args.model}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Image size   : {args.imgsz}")
    print(f"  Batch size   : {args.batch}")
    print(f"  YAML         : {yaml_path}")
    print(f"  Output dir   : {output_dir}\n")

    model = YOLO(args.model)

    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        project=str(output_dir),
        name="train",
        seed=args.seed,
        device=args.device if args.device else None,
        exist_ok=True,
        # Augmentation tweaks suitable for workplace face detection
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        degrees=5.0,
        scale=0.3,
        translate=0.1,
        # Cosine LR schedule
        cos_lr=True,
        warmup_epochs=3,
    )

    # ── 4. Save summary ──────────────────────────────────────────────────────
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    last_weights = Path(results.save_dir) / "weights" / "last.pt"

    summary = {
        "model": args.model,
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "results_dir": str(results.save_dir),
        "yaml": yaml_path,
    }

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[train_yolo] Training complete!")
    print(f"  Best weights : {best_weights}")
    print(f"  Summary      : {summary_path}")
    print(f"\nNext step — evaluate:")
    print(f"  python evaluate.py --weights {best_weights}")


if __name__ == "__main__":
    main()
