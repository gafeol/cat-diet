#!/usr/bin/env python3
"""Pi Camera + int8 TFLite: detect any cat in frame and save raw frames for dataset collection with logging."""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
from PIL import Image
from picamera2 import Picamera2

import common

MODEL = "mobilenet_v1_1.0_224_quant.tflite"
LABELS = "labels_mobilenet_quant_v1_224.txt"
RES = (1280, 720)
INPUT_SIZE = 224
SAVE_DIR = "captures"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
common.load_env(os.path.join(_SCRIPT_DIR, ".env"))
common.load_env(os.path.join(os.path.dirname(_SCRIPT_DIR), ".env"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=int, default=-1, help="frames to classify, -1 = forever"
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="min seconds between captures (avoid saving too many similar frames)",
    )
    args = parser.parse_args()

    # Setup logging
    log_file = os.path.join(_SCRIPT_DIR, "capture_data.log")
    common.setup_logging(log_path=log_file, level="INFO")
    logging.info("capture_data starting (pid=%s)", os.getpid())

    labels = common.load_labels(os.path.join(_SCRIPT_DIR, LABELS))
    interpreter, in_d, out_d = common.load_interpreter(
        os.path.join(_SCRIPT_DIR, MODEL), num_threads=1
    )
    logging.info(
        "input : %s dtype=%s quant=%s",
        in_d["shape"],
        in_d["dtype"],
        in_d["quantization"],
    )
    logging.info(
        "output: %s dtype=%s quant=%s",
        out_d["shape"],
        out_d["dtype"],
        out_d["quantization"],
    )

    cam = common.start_camera(RES[0], RES[1])

    os.makedirs(SAVE_DIR, exist_ok=True)
    state = common.load_json(os.path.join(SAVE_DIR, ".state.json"), {})
    total_captured = state.get("total_captured", 0)
    last_capture = 0.0
    run = 0

    try:
        while args.runs < 0 or run < args.runs:
            frame = cam.capture_array("main")
            arr = common.image_to_input(frame, INPUT_SIZE)

            t0 = time.perf_counter()
            out = common.run_inference(interpreter, in_d, out_d, arr)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            label, score, _ = common.top_result(out, labels)
            logging.info(
                "run %d: %6.0f ms  -> %s (%.2f)", run, elapsed_ms, label, score
            )

            if common.is_cat(label) and score >= args.threshold:
                if time.time() - last_capture >= args.interval:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(SAVE_DIR, f"{ts}_{label}_{score:.2f}.jpg")
                    Image.fromarray(frame).save(path, quality=85)
                    logging.info("saved: %s", path)
                    last_capture = time.time()
                    total_captured += 1
                    common.save_json(
                        os.path.join(SAVE_DIR, ".state.json"),
                        {"total_captured": total_captured},
                    )

            run += 1
    finally:
        cam.stop()
        cam.close()


if __name__ == "__main__":
    main()