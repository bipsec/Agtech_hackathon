#!/usr/bin/env python3
"""Resize the full-res 0626_RGB image to a smaller preview for the viewer."""

import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SCALE = 0.3
INPUT = os.path.join("Tif", "0626_RGB.tif")
OUTPUT = os.path.join("input", "0626_RGB_resized.jpg")


def main():
    print(f"Loading {INPUT} ...")
    img = Image.open(INPUT)
    w, h = img.size
    print(f"  Original : {w} x {h}  ({os.path.getsize(INPUT)/(1024*1024):.1f} MB)")

    new_w, new_h = int(w * SCALE), int(h * SCALE)
    print(f"  Resizing to {new_w} x {new_h}  (scale={SCALE}) ...")
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    resized.save(OUTPUT, "JPEG", quality=85)
    print(f"  Saved : {OUTPUT}  ({os.path.getsize(OUTPUT)/(1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
