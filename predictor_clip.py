#!/usr/bin/env python3
"""
CLIP-based failed-print predictor for F3DPD - Optimized for Raspberry Pi 4B+

Uses the fine-tuned CLIP classifier (best_print_classifier_t4.pth) for inference.
Optimizations for RPi 4B+:
- CPU-only inference with float32
- Lazy model loading
- Minimal memory footprint
- 4-thread torch operations (matches RPi 4B+ cores)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# RPi 4B+ optimizations
torch.set_num_threads(4)  # RPi 4B+ has 4 cores
torch.set_grad_enabled(False)

_MODEL: Optional["CLIPPredictor"] = None
_DEVICE = torch.device("cpu")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "best_print_classifier_t4.pth")
CLIP_CACHE_DIR = os.path.join(_SCRIPT_DIR, ".clip_cache")


class PrintClassifier(nn.Module):
    """Classifier head matching the training architecture."""
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class CLIPPredictor:
    """Lazy-loaded CLIP predictor optimized for RPi 4B+."""
    
    def __init__(self):
        self.clip_model = None
        self.preprocess = None
        self.classifier = None
        self._loaded = False
    
    def _load(self) -> None:
        if self._loaded:
            return
        
        logger.info("Loading CLIP model for RPi 4B+ (this may take a moment)...")
        
        try:
            import clip
        except ImportError:
            raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")
        
        # Use local cache directory to avoid permission issues and network downloads at boot
        os.makedirs(CLIP_CACHE_DIR, exist_ok=True)
        
        # Point CLIP to use our local cache (CLIP uses ~/.cache/clip by default)
        local_model_path = os.path.join(CLIP_CACHE_DIR, "ViT-B-32.pt")
        
        if os.path.exists(local_model_path):
            # Load from local cache (no network needed)
            logger.info(f"Loading CLIP from local cache: {local_model_path}")
            self.clip_model, self.preprocess = clip.load(local_model_path, device=_DEVICE, jit=False)
        else:
            # Fallback: download with retry (only on first run)
            logger.warning(f"CLIP model not found at {local_model_path}, downloading...")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.clip_model, self.preprocess = clip.load("ViT-B/32", device=_DEVICE, jit=False)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)
                        logger.warning(f"CLIP load failed (attempt {attempt+1}), retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"Failed to load CLIP after {max_retries} attempts: {e}")
        
        self.clip_model.eval()
        
        # Load classifier head
        self.classifier = PrintClassifier()
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        checkpoint = torch.load(MODEL_PATH, map_location=_DEVICE)
        
        # Extract classifier weights from checkpoint
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        classifier_state = {}
        for k, v in state_dict.items():
            if k.startswith("classifier."):
                classifier_state[k] = v
        
        self.classifier.load_state_dict(classifier_state)
        self.classifier.eval()
        self.classifier.to(_DEVICE)
        
        self._loaded = True
        logger.info("CLIP model loaded successfully")
    
    def predict(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Predict whether a print is failed given a BGR frame.
        
        Returns (is_failed, confidence) where confidence is the model's
        certainty in its prediction (0.0 to 1.0).
        """
        self._load()
        
        from PIL import Image
        
        # Convert BGR -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Preprocess and run through CLIP encoder
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(_DEVICE)
        
        with torch.no_grad():
            # Get CLIP image features
            features = self.clip_model.encode_image(image_tensor)
            features = features.float()
            
            # Run through classifier
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=-1)
        
        # Class 0 = success, Class 1 = failed
        success_prob = probs[0, 0].item()
        failed_prob = probs[0, 1].item()
        
        is_failed = failed_prob > success_prob
        confidence = failed_prob if is_failed else success_prob
        
        return is_failed, confidence


def _get_predictor() -> CLIPPredictor:
    global _MODEL
    if _MODEL is None:
        _MODEL = CLIPPredictor()
    return _MODEL


def predict_failed_from_bgr(frame_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Predict whether a print is failed given a BGR frame.
    
    Drop-in replacement for predictor.predict_failed_from_bgr().
    
    Returns (is_failed, score) where score is the confidence (0.0 to 1.0).
    For compatibility with existing code, score > 0.5 means successful when
    is_failed is False.
    """
    predictor = _get_predictor()
    is_failed, confidence = predictor.predict(frame_bgr)
    
    # Return score in same format as original predictor:
    # score represents "probability of success" for compatibility
    if is_failed:
        score = 1.0 - confidence  # Low score = failed
    else:
        score = confidence  # High score = success
    
    return is_failed, score
