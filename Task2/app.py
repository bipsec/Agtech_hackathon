#!/usr/bin/env python3
"""
CropScout Interactive Viewer

Workflow:
  1. Display the resized preview JPG with subplot boundaries
  2. Click on any subplot to select it
  3. Get bbox from CropScouts_Subplot_Full.json, scale to the TIF coordinate space
  4. Crop from the full-res TIF (or full-res JPG fallback)
  5. Run YOLO segmentation model  →  weed detection + corn count
  6. Display results: detections, segmentation overlay, statistics
"""

import json
import sys
import os
import csv
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from PIL import Image
from shapely.geometry import Point, Polygon as ShapelyPolygon
import cv2
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None

# YOLO inference settings
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class SubplotData:
    """One subplot annotation with coordinates in preview + full-res space."""

    def __init__(self, annotation, categories, preview_scale, crop_scale):
        self.id = annotation["id"]
        self.category_id = annotation["category_id"]
        self.category_name = next(
            (c["name"] for c in categories if c["id"] == self.category_id),
            "Unknown",
        )
        attrs = annotation.get("attributes", {})
        self.crop = attrs.get("crop", "Unknown")
        self.plot_num = attrs.get("plot_num", 0)
        self.subplot_id = attrs.get("subplot_id", 0)
        self.treatment = attrs.get("treatment", "Unknown")
        self.row_num = attrs.get("row_num", "?")
        self.col_num = attrs.get("col_num", "?")
        self.sub_num = attrs.get("sub_num", "?")

        seg = annotation["segmentation"][0]
        json_polygon = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]

        psx, psy = preview_scale
        self.preview_polygon = [(x * psx, y * psy) for x, y in json_polygon]

        bbox = annotation["bbox"]  # [x, y, w, h] in JSON-ref space
        csx, csy = crop_scale
        self.source_bbox = (
            bbox[0] * csx,
            bbox[1] * csy,
            (bbox[0] + bbox[2]) * csx,
            (bbox[1] + bbox[3]) * csy,
        )

        self.shapely_poly = ShapelyPolygon(self.preview_polygon)


# ---------------------------------------------------------------------------
# YOLO model wrapper
# ---------------------------------------------------------------------------

def get_bounding_box(mask):
    """Derive (x, y, width, height) from a binary mask's nonzero pixels."""
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


class WeedCornDetector:
    """
    YOLO segmentation inference following scripts/inference_corn_weeds.py:

    1. Write crop to a temp JPEG and pass the *path* to model.predict()
    2. Resize each raw mask to the original image size (INTER_NEAREST)
    3. Threshold at 0.5 → binary mask
    4. Derive bounding box (x, y, w, h) and area from the mask pixels
    5. Separate weed (class 0) and corn (class 1) detections
    """

    CLASS_WEED = 0
    CLASS_CORN = 1

    def __init__(self, model_path):
        print(f"[model] Loading {os.path.basename(model_path)} ...")
        self.model = YOLO(model_path)
        print(f"[model] Classes: {self.model.names}")

    def predict(self, image_rgb: np.ndarray):
        img_h, img_w = image_rgb.shape[:2]

        # --- run model via file path (matches reference script) ------------
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        try:
            os.close(tmp_fd)
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            results = self.model.predict(
                source=tmp_path,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
            )
        finally:
            os.unlink(tmp_path)

        result = results[0]

        weeds, corns = [], []
        weed_mask_combined = np.zeros((img_h, img_w), dtype=np.uint8)
        corn_mask_combined = np.zeros((img_h, img_w), dtype=np.uint8)

        if result.masks is not None:
            for mask_tensor, box, conf, cls in zip(
                result.masks.data,
                result.boxes.xyxy,
                result.boxes.conf,
                result.boxes.cls,
            ):
                class_id = int(cls.item())
                confidence = float(conf.item())

                seg_mask = mask_tensor.cpu().numpy()

                if seg_mask.shape != (img_h, img_w):
                    seg_mask = cv2.resize(
                        seg_mask,
                        (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                binary_mask = (seg_mask > 0.5).astype(np.uint8) * 255

                mask_bbox = get_bounding_box(binary_mask)
                if mask_bbox is None:
                    continue
                x, y, w, h = mask_bbox
                area = int(np.sum(binary_mask > 0))

                entry = dict(
                    bbox=(x, y, x + w, y + h),
                    x=x, y=y, width=w, height=h,
                    conf=confidence,
                    mask=binary_mask,
                    area=area,
                )

                if class_id == self.CLASS_WEED:
                    weeds.append(entry)
                    weed_mask_combined[binary_mask > 0] = 1
                elif class_id == self.CLASS_CORN:
                    corns.append(entry)
                    corn_mask_combined[binary_mask > 0] = 1

        overlay = self._make_overlay(image_rgb, weeds, corns)
        mask_vis = self._make_mask_vis(
            img_h, img_w, weed_mask_combined, corn_mask_combined, weeds, corns
        )

        return dict(
            weeds=weeds,
            corns=corns,
            weed_count=len(weeds),
            corn_count=len(corns),
            overlay=overlay,
            mask_vis=mask_vis,
        )

    @staticmethod
    def _make_overlay(image, weeds, corns):
        overlay = image.copy().astype(np.float64)
        for det in weeds:
            m = det["mask"] > 0
            overlay[m] = overlay[m] * 0.45 + np.array([255, 60, 60]) * 0.55
        for det in corns:
            m = det["mask"] > 0
            overlay[m] = overlay[m] * 0.45 + np.array([60, 180, 255]) * 0.55
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        for det in weeds:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 50, 50), 2)
        for det in corns:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 160, 255), 2)
        return overlay

    @staticmethod
    def _make_mask_vis(h, w, weed_mask, corn_mask, weeds, corns):
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[corn_mask > 0] = [60, 180, 255]
        vis[weed_mask > 0] = [255, 60, 60]

        for det in weeds:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 100), 2)
        for det in corns:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 200, 255), 2)
        return vis


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

CROP_COLORS = {
    "Buckwheat": "#FF6B6B",
    "Corn": "#FFD93D",
    "Dry beans": "#6BCB77",
    "Soybeans": "#4D96FF",
    "Wheat": "#FF8B3D",
}


class CropScoutViewer:
    """
    Parameters
    ----------
    preview_path  : resized image for the interactive viewer
    fullres_path  : full-resolution image for cropping (JPG or TIF)
    json_path     : COCO annotation file with subplot polygons / bboxes
    model_path    : YOLO segmentation model (.pt)
    """

    def __init__(self, preview_path, fullres_path, json_path, model_path):
        with open(json_path) as f:
            self.coco = json.load(f)

        img_info = self.coco["images"][0]
        self.json_w, self.json_h = img_info["width"], img_info["height"]

        print(f"[info] Loading preview: {preview_path}")
        self.preview_img = Image.open(preview_path)
        self.preview_arr = np.array(self.preview_img)
        self.preview_w, self.preview_h = self.preview_img.size

        self.fullres_path = fullres_path
        self._fullres_img = None

        tmp = Image.open(fullres_path)
        self.source_w, self.source_h = tmp.size
        tmp.close()

        preview_scale = (self.preview_w / self.json_w, self.preview_h / self.json_h)
        crop_scale = (self.source_w / self.json_w, self.source_h / self.json_h)

        self.subplots = [
            SubplotData(a, self.coco["categories"], preview_scale, crop_scale)
            for a in self.coco["annotations"]
        ]
        print(
            f"[info] {len(self.subplots)} subplots  |  "
            f"Preview {self.preview_w}x{self.preview_h}  |  "
            f"Source {self.source_w}x{self.source_h}  |  "
            f"JSON ref {self.json_w}x{self.json_h}"
        )

        self.detector = WeedCornDetector(model_path)
        self._highlight_patch = None

        self.output_dir = os.path.join(os.path.dirname(fullres_path), os.pardir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self._init_csvs()

        self._build_ui()

    @property
    def fullres_img(self):
        if self._fullres_img is None:
            print("[info] Loading full-res image (first click) ...")
            self._fullres_img = Image.open(self.fullres_path)
            print(f"[info] Full-res ready: {self._fullres_img.size}")
        return self._fullres_img

    # ---- CSV output -------------------------------------------------------

    def _init_csvs(self):
        self.weed_csv = os.path.join(self.output_dir, "weed_detections.csv")
        self.corn_csv = os.path.join(self.output_dir, "corn_counts.csv")

        with open(self.weed_csv, "w", newline="") as f:
            csv.writer(f).writerow(["x", "y", "width", "height", "area"])
        with open(self.corn_csv, "w", newline="") as f:
            csv.writer(f).writerow(["subplot_id", "count"])

        print(f"[csv] {self.weed_csv}")
        print(f"[csv] {self.corn_csv}")

    def _save_csvs(self, sp, res):
        if sp.crop == "Corn":
            with open(self.weed_csv, "a", newline="") as f:
                w = csv.writer(f)
                for det in res["weeds"]:
                    w.writerow([det["x"], det["y"], det["width"], det["height"], det["area"]])
            print(f"[csv] Saved {res['weed_count']} weed rows for subplot {sp.subplot_id}")

        with open(self.corn_csv, "a", newline="") as f:
            csv.writer(f).writerow([sp.subplot_id, res["corn_count"]])

        print(f"[csv] Saved corn count {res['corn_count']} for subplot {sp.subplot_id}")

    # ---- UI ---------------------------------------------------------------

    def _build_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(13, 13))
        self.fig.canvas.manager.set_window_title("CropScout Interactive Viewer")
        self.ax.imshow(self.preview_arr)
        self.ax.set_title(
            "CropScout Viewer  —  click any subplot to analyze",
            fontsize=13, fontweight="bold",
        )

        for sp in self.subplots:
            color = CROP_COLORS.get(sp.crop, "#FFFFFF")
            self.ax.add_patch(
                MplPolygon(sp.preview_polygon, closed=True, fill=False,
                           edgecolor=color, linewidth=1.4, alpha=0.75)
            )
            cx, cy = sp.shapely_poly.centroid.coords[0]
            self.ax.text(
                cx, cy, str(sp.subplot_id), fontsize=5, color=color,
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45),
            )

        self.ax.legend(
            handles=[Line2D([0], [0], color=c, lw=2, label=n) for n, c in CROP_COLORS.items()],
            loc="upper right", fontsize=9, framealpha=0.85,
        )
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        plt.tight_layout()

    # ---- Click handling ---------------------------------------------------

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        pt = Point(event.xdata, event.ydata)
        hit = next((s for s in self.subplots if s.shapely_poly.contains(pt)), None)
        if hit is None:
            print("[click] No subplot at that location.")
            return

        print(
            f"\n[click] Subplot {hit.subplot_id}  |  {hit.crop}  |  "
            f"{hit.treatment}  |  source bbox {tuple(int(v) for v in hit.source_bbox)}"
        )
        self._select(hit)

        cropped = self._crop(hit)
        if cropped is None:
            return

        print(f"[infer] Running YOLO on {cropped.shape[1]}x{cropped.shape[0]} crop ...")
        results = self.detector.predict(cropped)
        print(
            f"[infer] Weed: {results['weed_count']}   Corn: {results['corn_count']}"
        )
        self._save_csvs(hit, results)
        self._show_results(cropped, hit, results)

    def _select(self, sp):
        if self._highlight_patch is not None:
            self._highlight_patch.remove()
        self._highlight_patch = self.ax.add_patch(
            MplPolygon(sp.preview_polygon, closed=True, fill=True,
                       facecolor="yellow", edgecolor="red",
                       linewidth=2.5, alpha=0.30)
        )
        self.fig.canvas.draw_idle()

    # ---- Crop -------------------------------------------------------------

    def _crop(self, sp):
        x1, y1, x2, y2 = (int(v) for v in sp.source_bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.source_w, x2), min(self.source_h, y2)
        try:
            region = self.fullres_img.crop((x1, y1, x2, y2))
            arr = np.array(region)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            print(f"[crop] {x2-x1}x{y2-y1} px  ->  array {arr.shape}")
            return arr
        except Exception as exc:
            print(f"[error] Crop failed: {exc}")
            return None

    # ---- Results display --------------------------------------------------

    def _show_results(self, cropped, sp, res):
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.canvas.manager.set_window_title(
            f"Subplot {sp.subplot_id} — {sp.crop} / {sp.treatment}"
        )

        axes[0, 0].imshow(cropped)
        axes[0, 0].set_title(
            f"Subplot {sp.subplot_id}\n{sp.crop}  —  {sp.treatment}",
            fontsize=12, fontweight="bold",
        )
        axes[0, 0].axis("off")

        axes[0, 1].imshow(res["overlay"])
        axes[0, 1].set_title("Model Detections Overlay", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(res["mask_vis"])
        axes[1, 0].set_title(
            f"Segmentation Masks\n"
            f"Blue = Corn ({res['corn_count']})    Red = Weed ({res['weed_count']})",
            fontsize=11, fontweight="bold",
        )
        axes[1, 0].axis("off")

        self._draw_stats(axes[1, 1], sp, res)

        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show(block=False)

    @staticmethod
    def _draw_stats(ax, sp, res):
        ax.axis("off")

        weed_areas = [w["area"] for w in res["weeds"]]
        corn_areas = [c["area"] for c in res["corns"]]
        weed_confs = [w["conf"] for w in res["weeds"]]
        corn_confs = [c["conf"] for c in res["corns"]]

        lines = [
            f"Subplot ID    : {sp.subplot_id}",
            f"Crop          : {sp.crop}",
            f"Treatment     : {sp.treatment}",
            f"Plot / Row / Col : {sp.plot_num} / {sp.row_num} / {sp.col_num}",
            "",
            f"{'─' * 36}",
            f"  CORN  count        : {res['corn_count']}",
        ]
        if corn_areas:
            lines += [
                f"  CORN  total area   : {sum(corn_areas):,} px",
                f"  CORN  avg conf     : {np.mean(corn_confs):.2f}",
            ]
        lines += [
            "",
            f"  WEED  count        : {res['weed_count']}",
        ]
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

        text = "\n".join(lines)
        ax.text(
            0.05, 0.95, text, transform=ax.transAxes,
            fontsize=12, family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFFDE7",
                      edgecolor="#888", alpha=0.95),
        )

    # ---- run --------------------------------------------------------------

    def run(self):
        print("\n" + "=" * 58)
        print("  CropScout Interactive Viewer")
        print("  Left-click any subplot to run weed/corn analysis")
        print("=" * 58 + "\n")
        plt.show()


# ---------------------------------------------------------------------------
# Batch mode — process all subplots without GUI
# ---------------------------------------------------------------------------

def run_batch(fullres_path, json_path, model_path):
    with open(json_path) as f:
        coco = json.load(f)

    img_info = coco["images"][0]
    json_w, json_h = img_info["width"], img_info["height"]

    fullres_img = Image.open(fullres_path)
    source_w, source_h = fullres_img.size
    csx = source_w / json_w
    csy = source_h / json_h

    detector = WeedCornDetector(model_path)

    base = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base, "output")
    os.makedirs(output_dir, exist_ok=True)

    weed_csv = os.path.join(output_dir, "weed_detections.csv")
    corn_csv = os.path.join(output_dir, "corn_counts.csv")

    with open(weed_csv, "w", newline="") as f:
        csv.writer(f).writerow(["x", "y", "width", "height", "area"])
    with open(corn_csv, "w", newline="") as f:
        csv.writer(f).writerow(["subplot_id", "count"])

    categories = coco["categories"]
    annotations = coco["annotations"]
    total = len(annotations)

    print(f"\n[batch] Processing {total} subplots ...\n")

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

        results = detector.predict(crop_arr)

        with open(corn_csv, "a", newline="") as f:
            csv.writer(f).writerow([subplot_id, results["corn_count"]])

        if crop_name == "Corn":
            with open(weed_csv, "a", newline="") as f:
                w = csv.writer(f)
                for det in results["weeds"]:
                    w.writerow([det["x"], det["y"], det["width"], det["height"], det["area"]])

        weed_info = f"weed={results['weed_count']:>3d}  " if crop_name == "Corn" else " " * 14
        print(
            f"  [{i:>2}/{total}]  {subplot_id}  {crop_name:10s}  {treatment:10s}  "
            f"{weed_info}corn={results['corn_count']:>3d}"
        )

    print(f"\n[done] Results saved:")
    print(f"  {weed_csv}")
    print(f"  {corn_csv}\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    preview_path = os.path.join(base, "input", "0626_RGB_resized.jpg")
    json_path = os.path.join(base, "data", "CropScouts_Subplot_Full.json")
    model_path = os.path.join(base, "model", "weed_seg_last_20260221_151334.pt")

    tif_path = os.path.join(base, "Tif", "0626_RGB.tif")
    jpg_path = os.path.join(base, "input", "0626_RGB.jpg")

    if os.path.exists(tif_path):
        fullres_path = tif_path
    elif os.path.exists(jpg_path):
        fullres_path = jpg_path
    else:
        sys.exit(f"[error] No full-res source found at:\n  {tif_path}\n  {jpg_path}")

    for label, path in [("JSON", json_path), ("Model", model_path)]:
        if not os.path.exists(path):
            sys.exit(f"[error] {label} not found: {path}")

    batch = "--batch" in sys.argv

    if batch:
        print(f"[info] Full-res source: {fullres_path}")
        run_batch(fullres_path, json_path, model_path)
    else:
        if not os.path.exists(preview_path):
            sys.exit(f"[error] Preview not found: {preview_path}")
        print(f"[info] Full-res source: {fullres_path}")
        viewer = CropScoutViewer(preview_path, fullres_path, json_path, model_path)
        viewer.run()


if __name__ == "__main__":
    main()
