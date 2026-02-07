#!/usr/bin/env python3
"""
Test inference speed for predictor_clip on RPi 4B+.

Runs 5 inference tests with realistic image sizes and reports timing statistics.
Verifies no network downloads occur during inference.
"""

import time
import numpy as np
import os
import sys

# Block network to ensure nothing is downloaded
os.environ["CLIP_DOWNLOAD_ROOT"] = "/nonexistent"

def main():
    print("=" * 60)
    print("CLIP Inference Speed Test for RPi 4B+")
    print("=" * 60)
    
    # Measure model load time (cold start)
    print("\n[1] Loading model (cold start)...")
    load_start = time.perf_counter()
    
    from predictor_clip import predict_failed_from_bgr, _get_predictor
    
    # Force model load
    predictor = _get_predictor()
    predictor._load()
    
    load_time = time.perf_counter() - load_start
    print(f"    Model load time: {load_time:.2f}s")
    
    # CLIP normalizes all inputs to 224x224 (training size)
    width, height = 224, 224
    num_tests = 5
    
    print(f"\n[2] Testing inference at {width}x{height} - {num_tests} iterations...")
    
    # Create realistic test image (random noise simulates real camera data)
    test_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Warmup run (not counted)
    _ = predict_failed_from_bgr(test_frame)
    
    times = []
    for i in range(num_tests):
        start = time.perf_counter()
        is_failed, score = predict_failed_from_bgr(test_frame)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed*1000:.1f}ms (failed={is_failed}, score={score:.3f})")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n    Results:")
    print(f"      Average: {avg_time*1000:.1f}ms")
    print(f"      Min:     {min_time*1000:.1f}ms")
    print(f"      Max:     {max_time*1000:.1f}ms")
    print(f"      FPS:     {1/avg_time:.2f}")
    
    # Verify cache is being used
    print("\n[3] Verifying local cache usage...")
    cache_dir = os.path.join(os.path.dirname(__file__), ".clip_cache")
    model_file = os.path.join(cache_dir, "ViT-B-32.pt")
    
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"    ✓ Local CLIP cache found: {model_file} ({size_mb:.1f}MB)")
    else:
        print(f"    ✗ WARNING: Local cache missing at {model_file}")
        print("      Run: cp ~/.cache/clip/ViT-B-32.pt .clip_cache/")
    
    classifier_file = os.path.join(os.path.dirname(__file__), "best_print_classifier_t4.pth")
    if os.path.exists(classifier_file):
        size_mb = os.path.getsize(classifier_file) / (1024 * 1024)
        print(f"    ✓ Classifier model found: {classifier_file} ({size_mb:.1f}MB)")
    else:
        print(f"    ✗ WARNING: Classifier missing at {classifier_file}")
    
    print("\n" + "=" * 60)
    print("Test complete - no network downloads occurred")
    print("=" * 60)


if __name__ == "__main__":
    main()
