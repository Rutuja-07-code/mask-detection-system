"""Explore and visualise the raw mask-detection dataset.

Outputs:
  • Console summary table (class counts, bbox stats)
  • artifacts/exploration/sample_grid.png   — annotated sample images
  • artifacts/exploration/bbox_histogram.png — bounding-box size distribution
  • artifacts/exploration/class_balance.png  — per-split class bar chart
  • artifacts/exploration/dataset_stats.json — machine-readable stats

Usage::

    python explore_dataset.py
    python explore_dataset.py --dataset-root /path/to/raw/dataset --samples 16
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.mask_detection.constants import CLASS_COLORS, CLASS_NAMES, DEFAULT_RAW_DATASET_ROOT
from src.mask_detection.dataset_tools import load_image_records


# ---------------------------------------------------------------------------
# Colour map for matplotlib (CLASS_COLORS is BGR for OpenCV → convert to RGB float)
# ---------------------------------------------------------------------------
MPL_COLORS = {
    "with_mask":            "#2ecc71",   # green
    "without_mask":         "#e74c3c",   # red
    "mask_weared_incorrect": "#f1c40f",  # yellow
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore and visualise the mask dataset.")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_RAW_DATASET_ROOT,
        help="Raw dataset root with 'images' and 'annotations' sub-folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/exploration",
        help="Directory to write output images and JSON.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Number of sample images to draw in the annotation grid.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(records):
    """Return per-class annotation counts and bbox (w, h) pixel lists."""
    class_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    widths: dict[str, list[int]] = {name: [] for name in CLASS_NAMES}
    heights: dict[str, list[int]] = {name: [] for name in CLASS_NAMES}

    for record in records:
        for ann in record.annotations:
            label = ann.label
            class_counts[label] += 1
            xmin, ymin, xmax, ymax = ann.bbox
            widths[label].append(xmax - xmin)
            heights[label].append(ymax - ymin)

    return class_counts, widths, heights


def print_summary(records, class_counts, widths, heights) -> None:
    total = sum(class_counts.values())
    print("\n" + "=" * 60)
    print(f"  Smart Workplace Mask Compliance — Dataset Explorer")
    print("=" * 60)
    print(f"  Total images with annotations : {len(records)}")
    print(f"  Total face annotations        : {total}")
    print("-" * 60)
    print(f"  {'Class':<28} {'Count':>7}  {'% ':>6}  {'Avg W':>6}  {'Avg H':>6}")
    print("-" * 60)
    for name in CLASS_NAMES:
        cnt = class_counts[name]
        pct = 100 * cnt / total if total else 0
        avg_w = np.mean(widths[name]) if widths[name] else 0
        avg_h = np.mean(heights[name]) if heights[name] else 0
        print(f"  {name:<28} {cnt:>7}  {pct:>5.1f}%  {avg_w:>6.1f}  {avg_h:>6.1f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Plot: annotated image grid
# ---------------------------------------------------------------------------

def draw_annotated_grid(records, output_path: Path, n_samples: int, seed: int) -> None:
    rng = random.Random(seed)
    sample = rng.sample(records, min(n_samples, len(records)))

    cols = 4
    rows = math.ceil(len(sample) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), dpi=120)
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for ax, record in zip(axes, sample):
        img_bgr = cv2.imread(str(record.image_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        for ann in record.annotations:
            xmin, ymin, xmax, ymax = ann.bbox
            color_hex = MPL_COLORS.get(ann.label, "#ffffff")
            rect = plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor=color_hex, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                xmin, max(0, ymin - 4), ann.label.replace("_", " "),
                fontsize=6, color=color_hex,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, edgecolor="none"),
            )

        ax.imshow(img_rgb)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(record.image_path.name[:28], fontsize=7, pad=2)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=MPL_COLORS[n], label=n.replace("_", " ")) for n in CLASS_NAMES]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9, framealpha=0.8)

    fig.suptitle("Sample Annotated Images", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[explore] Saved annotation grid → {output_path}")


# ---------------------------------------------------------------------------
# Plot: bounding-box size histogram
# ---------------------------------------------------------------------------

def plot_bbox_histogram(widths, heights, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=130)

    for ax, data_map, xlabel in [
        (axes[0], widths, "Bounding-box Width (px)"),
        (axes[1], heights, "Bounding-box Height (px)"),
    ]:
        for name in CLASS_NAMES:
            vals = data_map[name]
            if vals:
                ax.hist(vals, bins=40, alpha=0.65, label=name.replace("_", " "),
                        color=MPL_COLORS[name], edgecolor="none")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Bounding-Box Size Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[explore] Saved bbox histogram    → {output_path}")


# ---------------------------------------------------------------------------
# Plot: class balance bar chart
# ---------------------------------------------------------------------------

def plot_class_balance(class_counts, output_path: Path) -> None:
    labels = [n.replace("_", " ") for n in CLASS_NAMES]
    counts = [class_counts[n] for n in CLASS_NAMES]
    colors = [MPL_COLORS[n] for n in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    bars = ax.bar(labels, counts, color=colors, edgecolor="none", linewidth=0)
    ax.set_ylabel("Annotation count", fontsize=11)
    ax.set_title("Class Balance — Full Dataset", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    total = sum(counts)
    for bar, cnt in zip(bars, counts):
        pct = 100 * cnt / total if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{cnt}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[explore] Saved class balance     → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[explore] Loading annotations from: {args.dataset_root}")
    records = load_image_records(args.dataset_root)
    class_counts, widths, heights = compute_stats(records)

    print_summary(records, class_counts, widths, heights)

    draw_annotated_grid(records, output_dir / "sample_grid.png", args.samples, args.seed)
    plot_bbox_histogram(widths, heights, output_dir / "bbox_histogram.png")
    plot_class_balance(class_counts, output_dir / "class_balance.png")

    # Save machine-readable stats
    stats = {
        "total_images": len(records),
        "total_annotations": sum(class_counts.values()),
        "class_counts": class_counts,
        "bbox_stats": {
            name: {
                "mean_width":  float(np.mean(widths[name]))  if widths[name]  else 0,
                "mean_height": float(np.mean(heights[name])) if heights[name] else 0,
                "min_width":   int(min(widths[name]))  if widths[name]  else 0,
                "max_width":   int(max(widths[name]))  if widths[name]  else 0,
                "min_height":  int(min(heights[name])) if heights[name] else 0,
                "max_height":  int(max(heights[name])) if heights[name] else 0,
            }
            for name in CLASS_NAMES
        },
    }
    stats_path = output_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[explore] Saved stats JSON        → {stats_path}")
    print(f"\n[explore] Done. All outputs in: {output_dir}\n")


if __name__ == "__main__":
    main()
