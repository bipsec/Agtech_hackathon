#!/usr/bin/env python3
"""
Batch weed detection + corn counting on all subplots.

Follows the inference pattern from scripts/inference_corn_weeds.py:
  model.predict(source=file_path) → resize masks (INTER_NEAREST)
  → binary threshold 0.5 → bbox & area from mask pixels

Outputs:
  output/weed_detections.csv   — x, y, width, height, area  (corn subplots only)
  output/corn_counts.csv       — subplot_id, count           (all subplots)
"""

import os
import sys
import csv
import json
import tempfile
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------------------
# Paths  (edit these if your layout differs)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "data", "CropScouts_Subplot_Full.json")
MODEL_PATH = os.path.join(BASE_DIR, "model", "weed_seg_last_20260221_151334.pt")
TIF_PATH = os.path.join(BASE_DIR, "Tif", "0626_RGB.tif")
JPG_PATH = os.path.join(BASE_DIR, "input", "0626_RGB.jpg")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Inference parameters
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45

# YOLO class IDs
CLASS_WEED = 0
CLASS_CORN = 1


# ---------------------------------------------------------------------------
# Helpers  (from scripts/inference_corn_weeds.py)
# ---------------------------------------------------------------------------

def get_bounding_box(mask):
    """(x, y, width, height) from a binary mask's nonzero pixels."""
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


def get_polygon_area(mask):
    """Total nonzero pixel count."""
    return int(np.sum(mask > 0))


# ---------------------------------------------------------------------------
# Per-subplot inference
# ---------------------------------------------------------------------------

def run_inference(crop_rgb, model):
    """
    Run YOLO on one subplot crop.

    Returns (weed_detections, corn_count) where each weed detection is
    a dict with x, y, width, height, area derived from the mask.
    """
    img_h, img_w = crop_rgb.shape[:2]

    # Write to temp file — YOLO's file-based pipeline gives better results
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.close(fd)
        bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        results = model.predict(
            source=tmp_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )
    finally:
        os.unlink(tmp_path)

    result = results[0]
    weeds = []
    corn_count = 0

    if result.masks is None:
        return weeds, corn_count

    for mask_tensor, conf, cls in zip(
        result.masks.data,
        result.boxes.conf,
        result.boxes.cls,
    ):
        class_id = int(cls.item())
        confidence = float(conf.item())

        # Resize mask to original crop dimensions
        seg_mask = mask_tensor.cpu().numpy()
        if seg_mask.shape != (img_h, img_w):
            seg_mask = cv2.resize(
                seg_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
            )

        # Binary mask
        binary_mask = (seg_mask > 0.5).astype(np.uint8) * 255

        if class_id == CLASS_CORN:
            corn_count += 1

        elif class_id == CLASS_WEED:
            bbox = get_bounding_box(binary_mask)
            if bbox is None:
                continue
            x, y, w, h = bbox
            area = get_polygon_area(binary_mask)
            weeds.append(dict(x=x, y=y, width=w, height=h, area=area, conf=confidence))

    return weeds, corn_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- resolve full-res source ---
    if os.path.exists(TIF_PATH):
        fullres_path = TIF_PATH
    elif os.path.exists(JPG_PATH):
        fullres_path = JPG_PATH
    else:
        sys.exit(f"No full-res image found at:\n  {TIF_PATH}\n  {JPG_PATH}")

    for label, path in [("JSON", JSON_PATH), ("Model", MODEL_PATH)]:
        if not os.path.exists(path):
            sys.exit(f"{label} not found: {path}")

    # --- load model ---
    print(f"Loading model: {os.path.basename(MODEL_PATH)}")
    model = YOLO(MODEL_PATH)
    print(f"Classes: {model.names}")

    # --- load annotations ---
    with open(JSON_PATH) as f:
        coco = json.load(f)
    json_w = coco["images"][0]["width"]
    json_h = coco["images"][0]["height"]

    # --- load full-res image ---
    print(f"Loading image: {fullres_path}")
    fullres = Image.open(fullres_path)
    src_w, src_h = fullres.size
    scale_x = src_w / json_w
    scale_y = src_h / json_h
    print(f"  Image {src_w}x{src_h}  |  JSON ref {json_w}x{json_h}  |  scale {scale_x:.4f}")

    # --- prepare output CSVs ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    weed_csv_path = os.path.join(OUTPUT_DIR, "weed_detections.csv")
    corn_csv_path = os.path.join(OUTPUT_DIR, "corn_counts.csv")

    weed_file = open(weed_csv_path, "w", newline="")
    corn_file = open(corn_csv_path, "w", newline="")
    weed_writer = csv.writer(weed_file)
    corn_writer = csv.writer(corn_file)
    weed_writer.writerow(["x", "y", "width", "height", "area"])
    corn_writer.writerow(["subplot_id", "count"])

    # --- process every subplot ---
    annotations = coco["annotations"]
    total = len(annotations)
    total_weeds = 0
    total_corns = 0

    print(f"\nProcessing {total} subplots ...\n")
    print(f"  {'#':>3}  {'subplot':>7}  {'crop':10}  {'treatment':10}  {'weed':>5}  {'corn':>5}")
    print(f"  {'─'*3}  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*5}  {'─'*5}")

    for i, ann in enumerate(annotations, 1):
        attrs = ann.get("attributes", {})
        crop_name = attrs.get("crop", "Unknown")
        subplot_id = attrs.get("subplot_id", ann["id"])
        treatment = attrs.get("treatment", "?")

        # Scale bbox from JSON space to source image space
        bbox = ann["bbox"]
        x1 = max(0, int(bbox[0] * scale_x))
        y1 = max(0, int(bbox[1] * scale_y))
        x2 = min(src_w, int((bbox[0] + bbox[2]) * scale_x))
        y2 = min(src_h, int((bbox[1] + bbox[3]) * scale_y))

        crop_arr = np.array(fullres.crop((x1, y1, x2, y2)))
        if crop_arr.ndim == 2:
            crop_arr = cv2.cvtColor(crop_arr, cv2.COLOR_GRAY2RGB)
        elif crop_arr.shape[2] == 4:
            crop_arr = crop_arr[:, :, :3]

        weeds, corn_count = run_inference(crop_arr, model)

        # Write corn count for every subplot
        corn_writer.writerow([subplot_id, corn_count])
        total_corns += corn_count

        # Write weed detections only for corn subplots
        weed_count = 0
        if crop_name == "Corn":
            for det in weeds:
                weed_writer.writerow([det["x"], det["y"], det["width"], det["height"], det["area"]])
            weed_count = len(weeds)
            total_weeds += weed_count

        weed_str = f"{weed_count:>5}" if crop_name == "Corn" else "    -"
        print(f"  {i:>3}  {subplot_id:>7}  {crop_name:10}  {treatment:10}  {weed_str}  {corn_count:>5}")

    weed_file.close()
    corn_file.close()

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"  Done — {total} subplots processed")
    print(f"  Total corn detections : {total_corns}")
    print(f"  Total weed detections : {total_weeds}  (corn subplots only)")
    print(f"{'='*60}")
    print(f"\n  {weed_csv_path}")
    print(f"  {corn_csv_path}\n")


if __name__ == "__main__":
    main()
