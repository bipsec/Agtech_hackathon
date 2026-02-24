#!/usr/bin/env python3
"""
Run weed detection inference only on corn subplot areas
Saves weed detections with coordinates, dimensions, and area to JSON
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# Configuration
IMAGE_PATH = "/home/test/ndsu agtech hackthon/task02/weed-crop/dataset/images/train/0710_RGB.jpg"
CORN_MASK_PATH = "/home/test/ndsu agtech hackthon/task02/masks/0710_RGB_corn_mask.png"
MODEL_DIR = "/home/test/ndsu agtech hackthon/task02/models"
MODEL_PATH = "/home/test/ndsu agtech hackthon/task02/models/weed_seg_last_20260221_151334.pt"  # Specific model to use
OUTPUT_DIR = "/home/test/ndsu agtech hackthon/task02/corn_weed_detections"

# Inference parameters
CONFIDENCE_THRESHOLD = 0.15  # Lowered from 0.25 to detect more weeds
IOU_THRESHOLD = 0.45

def find_latest_model(model_dir):
    """Find the most recent best model"""
    model_files = list(Path(model_dir).glob("weed_seg_best_*.pt"))
    if not model_files:
        return None
    
    # Sort by modification time, newest first
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    return str(latest_model)

def load_model(model_path=None):
    """Load YOLO segmentation model"""
    if model_path is None:
        model_path = MODEL_PATH if MODEL_PATH else find_latest_model(MODEL_DIR)
    
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    print(f"📦 Loading model: {os.path.basename(model_path)}")
    model = YOLO(model_path)
    print(f"✓ Model loaded successfully")
    
    return model

def load_corn_mask(mask_path):
    """Load the corn subplot mask"""
    print(f"\n📍 Loading corn mask: {os.path.basename(mask_path)}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise FileNotFoundError(f"Could not load corn mask from {mask_path}")
    
    # Ensure binary mask (0 or 255)
    mask = (mask > 127).astype(np.uint8) * 255
    
    corn_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    percentage = (corn_pixels / total_pixels) * 100
    
    print(f"✓ Corn mask loaded: {mask.shape[1]}×{mask.shape[0]}")
    print(f"  → Corn area coverage: {percentage:.2f}% ({corn_pixels:,} pixels)")
    
    return mask

def get_polygon_area(mask):
    """Calculate area of a segmentation mask in pixels"""
    return np.sum(mask > 0)

def get_bounding_box(mask):
    """
    Get bounding box from binary mask
    Returns: (x, y, width, height) where (x,y) is top-left corner
    """
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    x = int(x_min)
    y = int(y_min)
    width = int(x_max - x_min + 1)
    height = int(y_max - y_min + 1)
    
    return x, y, width, height

def get_weed_in_corn_area(weed_mask, corn_mask):
    """
    Calculate the area of weed that falls within corn regions
    
    Args:
        weed_mask: Binary mask of the weed detection
        corn_mask: Binary mask of corn areas
    
    Returns:
        (weed_in_corn_mask, area_in_corn, total_weed_area, overlap_ratio)
    """
    weed_pixels = weed_mask > 0
    corn_pixels = corn_mask > 0
    
    # Get intersection: weed pixels that are also in corn areas
    weed_in_corn = np.logical_and(weed_pixels, corn_pixels).astype(np.uint8) * 255
    
    area_in_corn = np.sum(weed_in_corn > 0)
    total_weed_area = np.sum(weed_pixels)
    
    overlap_ratio = area_in_corn / total_weed_area if total_weed_area > 0 else 0.0
    
    return weed_in_corn, area_in_corn, total_weed_area, overlap_ratio

def run_inference_on_corn(image_path, corn_mask, model, conf_threshold=0.25):
    """
    Run weed detection inference on entire image, then calculate area only for weed pixels in corn regions
    
    Returns:
        List of weed detections with area calculated only for pixels within corn areas
    """
    print(f"\n🔍 Running weed detection inference on entire image...")
    print(f"  → Confidence threshold: {conf_threshold}")
    print(f"  → Image: {os.path.basename(image_path)}")
    
    # Load image
    Image.MAX_IMAGE_PIXELS = None
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_height, img_width = image.shape[:2]
    print(f"  → Image size: {img_width}×{img_height}")
    
    # Run inference on entire image
    print(f"\n⏳ Processing entire image with model (this may take a while for large images)...")
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=IOU_THRESHOLD,
        verbose=False
    )
    
    result = results[0]
    
    # Process all weed detections
    detections_in_corn = []
    total_detections = 0
    weed_detections = 0
    weeds_touching_corn = 0
    
    if result.masks is not None:
        print(f"\n📊 Analyzing detections and calculating areas in corn regions...")
        
        for idx, (mask, box, conf, cls) in enumerate(zip(
            result.masks.data,
            result.boxes.xyxy,
            result.boxes.conf,
            result.boxes.cls
        )):
            total_detections += 1
            class_id = int(cls.item())
            
            # Only process weed class (class 0)
            if class_id != 0:
                continue
            
            weed_detections += 1
            
            # Convert mask to numpy array
            weed_mask = mask.cpu().numpy()
            
            # Resize mask to match image dimensions
            if weed_mask.shape != (img_height, img_width):
                weed_mask = cv2.resize(
                    weed_mask, 
                    (img_width, img_height),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Convert to binary mask
            weed_mask = (weed_mask > 0.5).astype(np.uint8) * 255
            
            # Get full weed bounding box (from entire weed mask)
            full_bbox = get_bounding_box(weed_mask)
            total_weed_area = get_polygon_area(weed_mask)
            
            # Calculate intersection with corn areas
            weed_in_corn_mask, area_in_corn, total_area, overlap_ratio = get_weed_in_corn_area(
                weed_mask, corn_mask
            )
            
            # Only include if there's any overlap with corn areas
            if area_in_corn > 0:
                weeds_touching_corn += 1
                
                # Get bounding box of the weed portion that's in corn
                corn_bbox = get_bounding_box(weed_in_corn_mask)
                
                if full_bbox is not None:
                    x_full, y_full, width_full, height_full = full_bbox
                    
                    # Use corn-specific bbox if available, otherwise use full bbox
                    if corn_bbox is not None:
                        x, y, width, height = corn_bbox
                    else:
                        x, y, width, height = full_bbox
                    
                    detections_in_corn.append({
                        'id': len(detections_in_corn) + 1,
                        'class': 'weed',
                        'confidence': float(conf.item()),
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'area_in_corn_pixels': int(area_in_corn),
                        'total_weed_area_pixels': int(total_weed_area),
                        'overlap_ratio': float(overlap_ratio),
                        'bbox_center_x': x + width // 2,
                        'bbox_center_y': y + height // 2,
                        'full_weed_bbox': {
                            'x': x_full,
                            'y': y_full,
                            'width': width_full,
                            'height': height_full
                        }
                    })
    
    print(f"\n✓ Inference complete!")
    print(f"  → Total detections: {total_detections}")
    print(f"  → Weed detections: {weed_detections}")
    print(f"  → Weeds with pixels in corn areas: {weeds_touching_corn}")
    print(f"  → Total area calculated for corn regions only")
    
    return detections_in_corn, image

def visualize_detections(image, detections, corn_mask, output_path):
    """Create visualization of weed detections in corn areas"""
    print(f"\n🎨 Creating visualization...")
    
    # Create overlay
    overlay = image.copy()
    
    # First, show corn areas in semi-transparent yellow
    corn_overlay = np.zeros_like(image)
    corn_overlay[corn_mask > 0] = [0, 255, 255]  # Yellow for corn
    overlay = cv2.addWeighted(overlay, 1.0, corn_overlay, 0.15, 0)
    
    # Draw each weed detection
    for det in detections:
        x, y, width, height = det['x'], det['y'], det['width'], det['height']
        
        # Draw bounding box (red for weeds)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 255), 3)
        
        # Draw label with area info
        area_corn = det['area_in_corn_pixels']
        label = f"Weed #{det['id']}: {det['confidence']:.2f} | {area_corn:,}px"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Background for text
        cv2.rectangle(overlay, 
                     (x, y - label_size[1] - 10), 
                     (x, y),
                     (0, 0, 255), -1)
        
        # Text
        cv2.putText(overlay, label, (x + 2, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save visualization
    cv2.imwrite(output_path, overlay)
    print(f"✓ Visualization saved: {output_path}")
    
    return overlay

def save_detections_json(detections, output_path, metadata):
    """Save detections to JSON file"""
    print(f"\n💾 Saving results to JSON...")
    
    output_data = {
        'metadata': metadata,
        'summary': {
            'total_weeds_in_corn': len(detections),
            'total_weed_area_in_corn_pixels': sum(d['area_in_corn_pixels'] for d in detections),
            'average_weed_area_in_corn_pixels': int(np.mean([d['area_in_corn_pixels'] for d in detections])) if detections else 0,
            'total_weed_area_all_pixels': sum(d['total_weed_area_pixels'] for d in detections),
        },
        'detections': detections
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ JSON saved: {output_path}")
    print(f"  → Total weeds touching corn: {len(detections)}")
    print(f"  → Total weed area in corn: {sum(d['area_in_corn_pixels'] for d in detections):,} pixels")
    
    return output_data

def main():
    print("="*70)
    print("WEED DETECTION IN CORN SUBPLOTS")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model = load_model()
    
    # Load corn mask
    if not os.path.exists(CORN_MASK_PATH):
        print(f"\n❌ Error: Corn mask not found at {CORN_MASK_PATH}")
        print(f"   Please run create_subplot_masks.py first to generate subplot masks.")
        return
    
    corn_mask = load_corn_mask(CORN_MASK_PATH)
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"\n❌ Error: Image not found at {IMAGE_PATH}")
        return
    
    # Run inference
    detections, image = run_inference_on_corn(
        IMAGE_PATH, 
        corn_mask, 
        model, 
        conf_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Prepare metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'image_path': IMAGE_PATH,
        'image_name': os.path.basename(IMAGE_PATH),
        'corn_mask_path': CORN_MASK_PATH,
        'model_path': MODEL_PATH if MODEL_PATH else find_latest_model(MODEL_DIR),
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD,
        'image_dimensions': {
            'width': image.shape[1],
            'height': image.shape[0]
        }
    }
    
    # Save results
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    json_path = os.path.join(OUTPUT_DIR, f"{base_name}_corn_weeds_{timestamp}.json")
    save_detections_json(detections, json_path, metadata)
    
    # Create visualization
    viz_path = os.path.join(OUTPUT_DIR, f"{base_name}_corn_weeds_viz_{timestamp}.jpg")
    visualize_detections(image, detections, corn_mask, viz_path)
    
    # Print summary
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nFiles generated:")
    print(f"  1. {os.path.basename(json_path)} - Detection data")
    print(f"  2. {os.path.basename(viz_path)} - Visualization")
    
    if detections:
        print(f"\nWeed Statistics in Corn Areas:")
        print(f"  → Total weeds detected: {len(detections)}")
        areas_in_corn = [d['area_in_corn_pixels'] for d in detections]
        total_areas = [d['total_weed_area_pixels'] for d in detections]
        print(f"  → Average weed area in corn: {np.mean(areas_in_corn):,.0f} pixels")
        print(f"  → Median weed area in corn: {np.median(areas_in_corn):,.0f} pixels")
        print(f"  → Min weed area in corn: {np.min(areas_in_corn):,} pixels")
        print(f"  → Max weed area in corn: {np.max(areas_in_corn):,} pixels")
        print(f"  → Total weed coverage in corn: {np.sum(areas_in_corn):,} pixels")
        print(f"  → Total weed area (all): {np.sum(total_areas):,} pixels")
        print(f"  → Average overlap with corn: {np.mean([d['overlap_ratio'] for d in detections])*100:.1f}%")
    else:
        print(f"\n⚠️  No weeds detected in corn areas")
    
    print("\n")

if __name__ == "__main__":
    main()
