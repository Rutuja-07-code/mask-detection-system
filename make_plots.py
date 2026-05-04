"""Generate plots from YOLOv8 training artifacts.

Reads the ``results.csv`` file that Ultralytics writes during training and
produces publication-ready PNG figures under ``artifacts/plots/`` (default).

Also supports the legacy ``training_summary.json`` from the old TensorFlow
pipeline if that file is present.

Usage::

    python make_plots.py
    python make_plots.py --results-csv artifacts/yolo/train/results.csv
    python make_plots.py --eval-json artifacts/evaluation/evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _apply_style() -> None:
    """Apply a consistent dark-ish scientific style."""
    plt.rcParams.update({
        "figure.facecolor":  "#0d0f14",
        "axes.facecolor":    "#161a23",
        "axes.edgecolor":    "#262d3f",
        "axes.labelcolor":   "#e2e8f4",
        "xtick.color":       "#7a8aaa",
        "ytick.color":       "#7a8aaa",
        "text.color":        "#e2e8f4",
        "grid.color":        "#262d3f",
        "legend.facecolor":  "#1e2433",
        "legend.edgecolor":  "#262d3f",
        "font.family":       "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 1. YOLOv8 training curves from results.csv
# ──────────────────────────────────────────────────────────────────────────────

def plot_yolo_training_curves(results_csv: Path, output_dir: Path) -> list[Path]:
    """Read Ultralytics results.csv and produce loss + metric curves."""
    try:
        import pandas as pd
    except ImportError:
        print("[make_plots] pandas not installed — skipping YOLO training curves.")
        return []

    if not results_csv.exists():
        print(f"[make_plots] results.csv not found: {results_csv} — skipping.")
        return []

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()   # strip whitespace from header

    written: list[Path] = []

    # ── Loss curves ──────────────────────────────────────────────────────────
    loss_cols = {
        "box_loss": ("Box Loss", "#3b82f6"),
        "cls_loss": ("Class Loss", "#f1c40f"),
        "dfl_loss": ("DFL Loss",  "#e74c3c"),
    }
    train_cols = {c: f"train/{c}" for c in loss_cols if f"train/{c}" in df.columns}
    val_cols   = {c: f"val/{c}"   for c in loss_cols if f"val/{c}"   in df.columns}

    if train_cols:
        fig, axes = plt.subplots(1, len(train_cols), figsize=(5 * len(train_cols), 4), dpi=150)
        if len(train_cols) == 1:
            axes = [axes]

        for ax, (col, (title, color)) in zip(axes, loss_cols.items()):
            tc = train_cols.get(col)
            vc = val_cols.get(col)
            epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
            if tc:
                ax.plot(epochs, df[tc], label="Train", color=color, linewidth=2)
            if vc:
                ax.plot(epochs, df[vc], label="Val",   color=color, linewidth=2, linestyle="--", alpha=0.7)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        fig.suptitle("YOLOv8 Training Losses", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        out = output_dir / "yolo_losses.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        print(f"[make_plots] Saved: {out}")

    # ── mAP / Precision / Recall curves ─────────────────────────────────────
    metric_cols = {
        "metrics/mAP50(B)":    ("mAP@0.5",   "#2ecc71"),
        "metrics/mAP50-95(B)": ("mAP@0.5:0.95", "#3b82f6"),
        "metrics/precision(B)": ("Precision", "#f1c40f"),
        "metrics/recall(B)":    ("Recall",    "#e74c3c"),
    }
    available = {k: v for k, v in metric_cols.items() if k in df.columns}

    if available:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

        # mAP subplot
        for col in ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            if col in available:
                label, color = available[col]
                epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
                axes[0].plot(epochs, df[col], label=label, color=color, linewidth=2)
        axes[0].set_title("mAP", fontsize=11)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)

        # Precision / Recall subplot
        for col in ["metrics/precision(B)", "metrics/recall(B)"]:
            if col in available:
                label, color = available[col]
                axes[1].plot(epochs, df[col], label=label, color=color, linewidth=2)
        axes[1].set_title("Precision & Recall", fontsize=11)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)

        fig.suptitle("YOLOv8 Detection Metrics", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        out = output_dir / "yolo_metrics.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        print(f"[make_plots] Saved: {out}")

    return written


# ──────────────────────────────────────────────────────────────────────────────
# 2. Per-class mAP bar chart from evaluation_results.json
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_class_map(eval_json: Path, output_dir: Path) -> list[Path]:
    if not eval_json.exists():
        print(f"[make_plots] Eval JSON not found: {eval_json} — skipping per-class bar chart.")
        return []

    data = _load_json(eval_json)
    per_class = data.get("per_class", [])
    if not per_class:
        return []

    CLASS_COLORS_MAP = {
        "with_mask":             "#2ecc71",
        "without_mask":          "#e74c3c",
        "mask_weared_incorrect": "#f1c40f",
    }

    labels = [pc["class"].replace("_", "\n") for pc in per_class]
    ap50   = [pc["ap50"]      for pc in per_class]
    prec   = [pc["precision"] for pc in per_class]
    rec    = [pc["recall"]    for pc in per_class]
    colors = [CLASS_COLORS_MAP.get(pc["class"], "#3b82f6") for pc in per_class]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    b1 = ax.bar(x - width, ap50, width, label="AP@0.5",    color=colors, alpha=0.9)
    b2 = ax.bar(x,         prec, width, label="Precision",  color=colors, alpha=0.6)
    b3 = ax.bar(x + width, rec,  width, label="Recall",     color=colors, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Class Detection Metrics (AP@0.5 / Precision / Recall)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    # Annotations
    for bar_group in [b1, b2, b3]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    overall = data.get("overall", {})
    summary = (
        f"Overall  mAP@0.5={overall.get('mAP@0.5', 0):.3f}  "
        f"P={overall.get('precision', 0):.3f}  "
        f"R={overall.get('recall', 0):.3f}  "
        f"FPS={overall.get('fps', 0):.1f}"
    )
    ax.set_xlabel(summary, fontsize=9, color="#7a8aaa")

    fig.tight_layout()
    out = output_dir / "per_class_metrics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_plots] Saved: {out}")
    return [out]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Legacy TensorFlow training curves (kept for backward compat)
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(summary: dict, output_dir: Path) -> list[Path]:
    history = summary.get("history", [])
    if not history:
        return []

    epochs   = [row["epoch"]          for row in history]
    train_loss = [row["train_loss"]   for row in history]
    val_loss   = [row["val_loss"]     for row in history]
    train_acc  = [row["train_accuracy"] for row in history]
    val_acc    = [row["val_accuracy"]   for row in history]
    lr         = [row.get("learning_rate") for row in history]

    written: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    axes[0].plot(epochs, train_loss, label="Train", color="#3b82f6", linewidth=2)
    axes[0].plot(epochs, val_loss,   label="Val",   color="#e74c3c", linewidth=2, linestyle="--")
    axes[0].set_title("Loss vs Epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train", color="#2ecc71", linewidth=2)
    axes[1].plot(epochs, val_acc,   label="Val",   color="#f1c40f", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy vs Epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3); axes[1].legend()

    fig.tight_layout()
    out = output_dir / "training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    written.append(out)
    print(f"[make_plots] Saved: {out}")

    if any(v is not None for v in lr):
        fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=150)
        ax.plot(epochs, lr, marker="o", color="#3b82f6")
        ax.set_title("Learning Rate vs Epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        out = output_dir / "learning_rate.png"
        fig.savefig(out); plt.close(fig)
        written.append(out)
        print(f"[make_plots] Saved: {out}")

    return written


def plot_class_distribution(summary: dict, output_dir: Path) -> list[Path]:
    dataset = summary.get("dataset") or {}
    splits  = dataset.get("splits") or {}
    if not splits:
        return []

    CLASS_COLORS_MAP = {"with_mask": "#2ecc71", "without_mask": "#e74c3c", "mask_weared_incorrect": "#f1c40f"}
    classes = dataset.get("classes") or sorted(
        {cls for s in splits.values() for cls in (s.get("class_counts") or {}).keys()}
    )

    split_names = ["train", "val", "test"]
    counts = {
        s: [int((splits.get(s, {}).get("class_counts") or {}).get(c, 0)) for c in classes]
        for s in split_names
    }

    x = np.arange(len(classes)); w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=150)
    ax.bar(x - w, counts["train"], w, label="Train", color="#3b82f6", alpha=0.85)
    ax.bar(x,     counts["val"],   w, label="Val",   color="#2ecc71", alpha=0.85)
    ax.bar(x + w, counts["test"],  w, label="Test",  color="#f1c40f", alpha=0.85)
    ax.set_title("Class Distribution (Face Crops)", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=0)
    ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    out = output_dir / "class_distribution.png"
    fig.savefig(out); plt.close(fig)
    print(f"[make_plots] Saved: {out}")
    return [out]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate plots for the Mask Compliance Monitor.")
    parser.add_argument(
        "--results-csv",
        default="artifacts/yolo/train/results.csv",
        help="Ultralytics results.csv from YOLO training.",
    )
    parser.add_argument(
        "--eval-json",
        default="artifacts/evaluation/evaluation_results.json",
        help="Evaluation results JSON (from evaluate.py).",
    )
    parser.add_argument(
        "--summary",
        default="artifacts/training_summary.json",
        help="Legacy TF training_summary.json (optional).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/plots",
        help="Directory where PNG plots will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _apply_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # YOLO training curves
    written.extend(plot_yolo_training_curves(Path(args.results_csv), output_dir))

    # Per-class evaluation bar chart
    written.extend(plot_per_class_map(Path(args.eval_json), output_dir))

    # Legacy TF curves (optional)
    tf_summary = Path(args.summary)
    if tf_summary.exists():
        summary = _load_json(tf_summary)
        written.extend(plot_training_curves(summary, output_dir))
        written.extend(plot_class_distribution(summary, output_dir))

    index = output_dir / "index.json"
    _save_json(index, {"written": [str(p) for p in written]})

    print(f"\n[make_plots] {len(written)} plot(s) saved to: {output_dir}")
    if not written:
        print("[make_plots] No data found. Run train_yolo.py and evaluate.py first.")


if __name__ == "__main__":
    main()
