#!/usr/bin/env python3
"""Shared helpers for cat-diet scripts: env loading, labels, cat detection, TFLite model loading, inference, JSON persistence, logging, and camera/image utilities."""

from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

import numpy as np
from PIL import Image

CAT_KEYWORDS = ("cat", "tabby", "lynx", "cougar", "jaguar", "leopard", "cheetah")

LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MiB per file
LOG_BACKUP_COUNT: int = 1  # total ≈ 5 + 5 = 10 MiB


def setup_logging(
    log_path: Optional[str] = None,
    level: str = "INFO",
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
) -> logging.Logger:
    """Configure root logger: console + optional rotating file.
    The file never exceeds *max_bytes*; after rotation *backup_count* old files are kept.
    Returns the logger (root) so every module inherits the handlers."""
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # --- console handler ---
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(ch)

    # --- optional rotating file ---
    if log_path:
        path = os.path.abspath(log_path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fh = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(fh)

    return logger


def load_env(path: str) -> None:
    """Load KEY=VALUE pairs from *path* into the environment (without overriding)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_labels(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


def is_cat(label: str) -> bool:
    low = label.lower()
    return any(k in low for k in CAT_KEYWORDS)


def load_interpreter(model_path: str, num_threads: int = 1):
    """Load an int8 TFLite model, returning (interpreter, input_details, output_details)."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter

            logging.getLogger(__name__).warning(
                "Using TensorFlow's lite interpreter (not tflite-runtime)"
            )
        except ImportError:
            raise ImportError(
                "No TFLite interpreter found. Install one:\n"
                "  Raspberry Pi: pip install tflite-runtime\n"
                "  Mac/Linux:    pip install tensorflow"
            )
    interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    return (
        interpreter,
        interpreter.get_input_details()[0],
        interpreter.get_output_details()[0],
    )


def run_inference(
    interpreter: Any, in_d: dict, out_d: dict, img_array: np.ndarray
) -> np.ndarray:
    """Run one inference on a pre-shaped input array; returns the raw output tensor."""
    interpreter.set_tensor(in_d["index"], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(out_d["index"])[0]


def top_result(out: np.ndarray, labels: list[str]) -> tuple[str, float, int]:
    """Return (label, score, index) for the argmax class, normalising uint8 output."""
    top = int(np.argmax(out))
    score = out[top] / 255.0 if out.dtype == np.uint8 else out[top]
    label = labels[top] if top < len(labels) else f"class_{top}"
    return label, score, top


def load_json(path: str, default: dict[str, Any]) -> dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, obj: Any) -> None:
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to save state: %s", e)


def get_logger(name: str = __name__) -> logging.Logger:
    return logging.getLogger(name)


# --- Camera & Image helpers ---

def start_camera(width: int, height: int, warmup: float = 2.0):
    """Initialize Picamera2 with RGB888 still configuration."""
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (width, height), "format": "RGB888"}))
    cam.start()
    time.sleep(warmup)
    return cam


def image_to_input(frame: np.ndarray, input_size: int) -> np.ndarray:
    """Resize frame to model input size and add batch axis (uint8)."""
    arr = np.expand_dims(
        np.asarray(
            Image.fromarray(frame).resize((input_size, input_size), Image.LANCZOS),
            dtype=np.uint8
        ),
        axis=0
    )
    return arr


def to_gray_small(frame: np.ndarray, det_w: int, det_h: int) -> np.ndarray:
    """Downscale frame to small grayscale float32 for change detection."""
    small = Image.fromarray(frame).resize((det_w, det_h), Image.BILINEAR).convert("L")
    return np.asarray(small, dtype=np.float32)


import time  # for start_camera warmup