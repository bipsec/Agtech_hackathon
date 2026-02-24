#!/usr/bin/env python3
"""
Batch-generate detection plots for all 60 subplots.

For each subplot, produces a 2x2 figure:
  [Original crop]  [Detection overlay]
  [Mask vis]        [Stats text]

Saves each as Task2/plots/<subplot_id>.jpg
Also saves individual panels:
  Task2/plots/<subplot_id>_crop.jpg
  Task2/plots/<subplot_id>_overlay.jpg
  Task2/plots/<subplot_id>_mask.jpg

Usage:
  python batch_generate.py
"""

import json
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import WeedCornDetector

Image.MAX_IMAGE_PIXELS = None

BASE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE, "data", "CropScouts_Subplot_Full.json")
MODEL_PATH = os.path.join(BASE, "model", "weed_seg_last_20260221_151334.pt")
TIF_PATH = os.path.join(BASE, "Tif", "0626_RGB.tif")
JPG_PATH = os.path.join(BASE, "input", "0626_RGB.jpg")
PLOTS_DIR = os.path.join(BASE, "plots")
CSV_INFO = os.path.join(BASE, "plots", "plot_index.json")


def draw_stats_text(ax, attrs, res):
    ax.axis("off")
    weed_areas = [w["area"] for w in res["weeds"]]
    corn_areas = [c["area"] for c in res["corns"]]
    weed_confs = [w["conf"] for w in res["weeds"]]
    corn_confs = [c["conf"] for c in res["corns"]]

    lines = [
        f"Subplot ID    : {attrs.get('subplot_id', '?')}",
        f"Crop          : {attrs.get('crop', '?')}",
        f"Treatment     : {attrs.get('treatment', '?')}",
        f"Plot / Row / Col : {attrs.get('plot_num','?')} / {attrs.get('row_num','?')} / {attrs.get('col_num','?')}",
        "",
        f"{'─' * 36}",
        f"  CORN  count        : {res['corn_count']}",
    ]
    if corn_areas:
        lines += [
            f"  CORN  total area   : {sum(corn_areas):,} px",
            f"  CORN  avg conf     : {np.mean(corn_confs):.2f}",
        ]
    lines += ["", f"  WEED  count        : {res['weed_count']}"]
    if weed_areas:
        lines += [
            f"  WEED  total area   : {sum(weed_areas):,} px",
            f"  WEED  avg conf     : {np.mean(weed_confs):.2f}",
        ]
    lines += [
        "",
        f"{'─' * 36}",
        f"  Total detections   : {res['corn_count'] + res['weed_count']}",
    ]

    ax.text(
        0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        fontsize=11, family="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFFDE7",
                  edgecolor="#888", alpha=0.95),
    )


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if os.path.exists(TIF_PATH):
        fullres_path = TIF_PATH
    elif os.path.exists(JPG_PATH):
        fullres_path = JPG_PATH
    else:
        sys.exit(f"[error] No source image at:\n  {TIF_PATH}\n  {JPG_PATH}")

    for label, path in [("JSON", JSON_PATH), ("Model", MODEL_PATH)]:
        if not os.path.exists(path):
            sys.exit(f"[error] {label} not found: {path}")

    with open(JSON_PATH) as f:
        coco = json.load(f)

    img_info = coco["images"][0]
    json_w, json_h = img_info["width"], img_info["height"]

    print(f"[info] Loading full-res: {fullres_path}")
    fullres_img = Image.open(fullres_path)
    source_w, source_h = fullres_img.size
    csx = source_w / json_w
    csy = source_h / json_h
    print(f"[info] Source {source_w}x{source_h}, JSON ref {json_w}x{json_h}, scale {csx:.4f}x{csy:.4f}")

    detector = WeedCornDetector(MODEL_PATH)

    annotations = coco["annotations"]
    total = len(annotations)
    index = []

    print(f"\n[batch] Generating plots for {total} subplots → {PLOTS_DIR}\n")

    for i, ann in enumerate(annotations, 1):
        attrs = ann.get("attributes", {})
        crop_name = attrs.get("crop", "Unknown")
        subplot_id = attrs.get("subplot_id", ann["id"])
        treatment = attrs.get("treatment", "?")

        bbox = ann["bbox"]
        x1 = max(0, int(bbox[0] * csx))
        y1 = max(0, int(bbox[1] * csy))
        x2 = min(source_w, int((bbox[0] + bbox[2]) * csx))
        y2 = min(source_h, int((bbox[1] + bbox[3]) * csy))

        crop_arr = np.array(fullres_img.crop((x1, y1, x2, y2)))
        if crop_arr.ndim == 2:
            crop_arr = cv2.cvtColor(crop_arr, cv2.COLOR_GRAY2RGB)
        elif crop_arr.shape[2] == 4:
            crop_arr = crop_arr[:, :, :3]

        res = detector.predict(crop_arr)

        sid = str(subplot_id)

        # Save individual panels
        Image.fromarray(crop_arr).save(
            os.path.join(PLOTS_DIR, f"{sid}_crop.jpg"), quality=85)
        Image.fromarray(res["overlay"]).save(
            os.path.join(PLOTS_DIR, f"{sid}_overlay.jpg"), quality=85)
        Image.fromarray(res["mask_vis"]).save(
            os.path.join(PLOTS_DIR, f"{sid}_mask.jpg"), quality=85)

        # Save 2x2 composite
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            f"Subplot {subplot_id} — {crop_name} / {treatment}",
            fontsize=14, fontweight="bold", y=0.98,
        )

        axes[0, 0].imshow(crop_arr)
        axes[0, 0].set_title(f"Original Crop\n{crop_name} — {treatment}", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(res["overlay"])
        axes[0, 1].set_title("Detection Overlay", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(res["mask_vis"])
        axes[1, 0].set_title(
            f"Segmentation Masks\nCorn={res['corn_count']}  Weed={res['weed_count']}",
            fontsize=10, fontweight="bold",
        )
        axes[1, 0].axis("off")

        draw_stats_text(axes[1, 1], attrs, res)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(PLOTS_DIR, f"{sid}.jpg"), dpi=120,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

        index.append({
            "subplot_id": subplot_id,
            "crop": crop_name,
            "treatment": treatment,
            "plot_num": attrs.get("plot_num", 0),
            "row": attrs.get("row_num", 0),
            "col": attrs.get("col_num", 0),
            "sub": attrs.get("sub_num", 0),
            "corn_count": res["corn_count"],
            "weed_count": res["weed_count"],
            "files": {
                "composite": f"{sid}.jpg",
                "crop": f"{sid}_crop.jpg",
                "overlay": f"{sid}_overlay.jpg",
                "mask": f"{sid}_mask.jpg",
            },
        })

        weed_info = f"weed={res['weed_count']:>3d}  " if crop_name == "Corn" else " " * 14
        print(
            f"  [{i:>2}/{total}]  {subplot_id}  {crop_name:10s}  {treatment:10s}  "
            f"{weed_info}corn={res['corn_count']:>3d}  ✓ saved"
        )

    with open(CSV_INFO, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n[done] {total} plots saved to {PLOTS_DIR}")
    print(f"[done] Index: {CSV_INFO}")


if __name__ == "__main__":
    main()
