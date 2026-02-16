#!/usr/bin/env python3
"""Quick test: run the same greyscale detection used by orchestrator on test_greyscale.jpg."""

import cv2
import numpy as np

GREYSCALE_SATURATION_THRESHOLD = 5

def is_frame_greyscale(frame_bgr: np.ndarray) -> bool:
    b, g, r = cv2.split(frame_bgr)
    b = b.astype(np.int16)
    g = g.astype(np.int16)
    r = r.astype(np.int16)
    max_ch = np.maximum(np.maximum(r, g), b)
    min_ch = np.minimum(np.minimum(r, g), b)
    mean_saturation = float(np.mean(max_ch - min_ch))
    return mean_saturation, mean_saturation < GREYSCALE_SATURATION_THRESHOLD

for name in ["test_greyscale.jpg", "greyscale_test_grey.jpg"]:
    img = cv2.imread(name)
    if img is None:
        print(f"ERROR: could not load {name}\n")
        continue
    mean_sat, is_grey = is_frame_greyscale(img)
    print(f"[{name}]")
    print(f"  Mean saturation : {mean_sat:.2f}")
    print(f"  Threshold       : {GREYSCALE_SATURATION_THRESHOLD}")
    print(f"  Greyscale?      : {is_grey}\n")
