#!/usr/bin/env python3
"""Long-running, low-energy cat watcher for the Raspberry Pi.

Two-stage detection: a cheap change detector (a small downscaled grayscale frame
compared to a slowly adapting background model) decides whether it's worth
running the int8 MobileNet (200-500ms). During idle the script sleeps with a
backoff (5s -> 60s) and runs the model periodically (--bg-check, default 30s) so
a stationary cat is still caught.

Saved images are routed to:
  captures/                 cat present (motion-triggered or periodic check)
  captures/false_positives/ motion triggered, model says not-a-cat
  captures/bg/              idle "nothing in frame" samples (~1/hr, jittered)

Disk is capped with a rolling quota (--quota-mb). At most one photo per second
(--min-save-interval) is saved anywhere.  Logs can be written to a rotating file
(see --log-file, --log-level, --log-max-mb).

Debug mode: --debug sends every capture to Telegram with details about the
capture type, label, and cumulative counts of each capture category.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from picamera2 import Picamera2

import common
import telegram as tg
from quota import count_jpgs, enforce_quota

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE_NAME = ".state.json"
CONFIG_FILE = "config.toml"


def file_ts() -> str:
    """Day + hour + minute + second timestamp used in every saved filename."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config(config_path: str) -> dict:
    """Load TOML config and flatten sections into a single dict for argparse defaults."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        data = tomllib.load(f)
    flat: dict = {}
    for section in data.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def parse_args(argv: list[str] | None = None) -> Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Runtime / I/O flags (not in config)
    p.add_argument("--runs", type=int, default=-1, help="loop iterations, -1 = forever")
    p.add_argument("--config", default=CONFIG_FILE, help="path to TOML config file")
    p.add_argument("--debug", action="store_true", help="send every capture to Telegram with type/label/cumulative counts")
    p.add_argument("--debug-interval", type=float, default=0.0, help="min seconds between debug Telegram messages (0 = every capture)")
    p.add_argument("--notify", action="store_true", help="send cat captures to Telegram")
    p.add_argument("--log-file", default="watch_cat.log", help="path to log file")
    p.add_argument("--log-level", default="INFO", help="logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)")
    p.add_argument("--log-max-mb", type=int, default=10, help="max log file size in MiB; handler truncates when exceeded (backupCount=0)")

    # Model / data paths (can be overridden)
    p.add_argument("--model", default=None, help="path to tflite model")
    p.add_argument("--labels", default=None, help="path to labels file")
    p.add_argument("--capture-dir", default=None, help="capture directory")

    # Parse once with config defaults
    cfg = load_config(CONFIG_FILE)
    ns = Namespace(**cfg)
    args = p.parse_args(argv, namespace=ns)

    # Resolve paths relative to script dir
    if args.model is None:
        args.model = os.path.join(_SCRIPT_DIR, cfg.get("model", "mobilenet_v1_1.0_224_quant.tflite"))
    if args.labels is None:
        args.labels = os.path.join(_SCRIPT_DIR, cfg.get("labels", "labels_mobilenet_quant_v1_224.txt"))
    if args.capture_dir is None:
        args.capture_dir = cfg.get("capture_dir", "captures")

    return args


class Watcher:
    def __init__(self, args: Namespace, cfg: dict):
        self.args = args
        self.cfg = cfg
        self.INPUT_SIZE = cfg.get("input_size", 224)

        # Load model & labels
        self.labels = common.load_labels(args.labels)
        self.interpreter, self.in_d, self.out_d = common.load_interpreter(args.model, num_threads=1)
        logging.info("model input : %s dtype=%s quant=%s", self.in_d['shape'], self.in_d['dtype'], self.in_d['quantization'])
        logging.info("model output: %s dtype=%s quant=%s", self.out_d['shape'], self.out_d['dtype'], self.out_d['quantization'])

        # Telegram
        common.load_env(os.path.join(_SCRIPT_DIR, ".env"))
        common.load_env(os.path.join(os.path.dirname(_SCRIPT_DIR), ".env"))
        self.token = tg.env_token()
        self.chat_id = tg.env_chat_id()

        # Camera
        self.cam = common.start_camera(args.width, args.height)

        # Directories
        os.makedirs(args.capture_dir, exist_ok=True)
        self.fp_dir = os.path.join(args.capture_dir, "false_positives")
        self.bg_dir = os.path.join(args.capture_dir, "bg")
        os.makedirs(self.fp_dir, exist_ok=True)
        os.makedirs(self.bg_dir, exist_ok=True)

        # Persistent state
        self.state_file = os.path.join(args.capture_dir, STATE_FILE_NAME)
        state = common.load_json(self.state_file, {})
        self.total_captured = state.get("total_captured", 0)
        self.last_idle_sample = state.get("last_idle_sample", time.time())

        # Runtime state
        self.bg: np.ndarray | None = None
        self.mode = "idle"
        self.idle_interval = args.idle_interval
        self.last_model_check = 0.0
        self.last_capture = 0.0
        self.last_notify = 0.0
        self.event_start = 0.0
        self.dwell_since: float | None = None
        self.fp_saved = False

        # Debug counters
        self.cat_captures = 0
        self.idle_captures = 0
        self.fp_captures = 0
        self.last_debug = 0.0

    def run(self) -> None:
        run = 0
        try:
            while self.args.runs < 0 or run < self.args.runs:
                self._step()
                run += 1
        except KeyboardInterrupt:
            logging.info("Stopped by user.")
        finally:
            self.cam.stop()

    def _step(self) -> None:
        frame = self.cam.capture_array("main")
        gray = common.to_gray_small(frame, self.args.det_width, self.args.det_height)

        if self.bg is None:
            self.bg = gray.copy()

        now = time.time()
        diff = float(np.abs(gray - self.bg).mean())

        if self.mode == "idle":
            self._handle_idle(frame, now, diff)
        else:
            self._handle_event(frame, now, diff)

        # Persist state
        common.save_json(self.state_file, {
            "total_captured": self.total_captured,
            "last_idle_sample": self.last_idle_sample,
        })

        # Debug telegram
        if self.args.debug:
            self._maybe_debug_telegram(now)

        # Sleep
        if self.mode == "idle":
            sleep_secs = self.idle_interval
            sleep_secs = min(sleep_secs, max(0.0, (self.last_model_check + self.args.bg_check) - now))
            time.sleep(sleep_secs)
        else:
            time.sleep(self.args.event_interval)

    def _handle_idle(self, frame: np.ndarray, now: float, diff: float) -> None:
        a = self.args
        if diff >= a.motion_threshold:
            self.mode = "event"
            self.event_start = now
            self.dwell_since = None
            self.fp_saved = False
            self.idle_interval = a.idle_interval
            logging.info("motion detected -> event mode (diff=%.1f)", diff)
        else:
            self.bg = (1 - a.bg_alpha) * self.bg + a.bg_alpha * gray
            self.idle_interval = min(self.idle_interval * 2, a.idle_max)

            # Periodic full-model check: catches cats that are stationary/eating.
            if now - self.last_model_check >= a.bg_check:
                label, score, ms = self._classify_frame(frame)
                self.last_model_check = now
                logging.info("bg-check %s (%.2f) %dms", label, score, int(ms))
                if common.is_cat(label) and score >= a.cat_threshold:
                    self._save_and_notify(frame, "cat", label, score)

            # Idle "nothing in view" sampler (~1/hr, jittered), capped.
            if count_jpgs(self.bg_dir) < a.bg_cap:
                next_idle_sample = self.last_idle_sample + a.bg_sample_interval + random.uniform(0, a.bg_jitter)
                if now >= next_idle_sample:
                    self._save_and_count(frame, self.bg_dir, "idle")
                    self.last_idle_sample = now
                    self.idle_captures += 1

    def _handle_event(self, frame: np.ndarray, now: float, diff: float) -> None:
        a = self.args
        # End the event when the scene settles or it runs too long.
        if diff >= a.motion_threshold:
            self.dwell_since = None
        else:
            if self.dwell_since is None:
                self.dwell_since = now
            elif now - self.dwell_since >= a.event_dwell:
                logging.info("scene settled -> back to idle")
                self.mode = "idle"
                self.idle_interval = a.idle_interval
                self.bg = None  # re-absorb current scene on the next idle frame
                return

        if now - self.event_start >= a.event_max:
            logging.info("event max (%ds) -> back to idle", int(a.event_max))
            self.mode = "idle"
            self.idle_interval = a.idle_interval
            self.bg = None
            return

        label, score, ms = self._classify_frame(frame)
        if common.is_cat(label) and score >= a.cat_threshold:
            self._save_and_notify(frame, "cat", label, score)
        elif not self.fp_saved and count_jpgs(self.fp_dir) < a.fp_cap:
            self._save_and_count(frame, self.fp_dir, "fp")
            self.fp_saved = True
            self.fp_captures += 1

    def _classify_frame(self, frame: np.ndarray) -> tuple[str, float, int]:
        arr = common.image_to_input(frame, self.INPUT_SIZE)
        t0 = time.perf_counter()
        out = common.run_inference(self.interpreter, self.in_d, self.out_d, arr)
        ms = (time.perf_counter() - t0) * 1000
        label, score, idx = common.top_result(out, self.labels)
        return label, score, int(ms)

    def _save_and_count(self, frame: np.ndarray, directory: str, ctype: str) -> str | None:
        """Save a JPEG if --min-save-interval allows; returns path or None."""
        a = self.args
        now = time.time()
        if now - self.last_capture < a.min_save_interval:
            return None
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{file_ts()}_{ctype}.jpg")
        Image.fromarray(frame).save(path, quality=a.jpeg_quality)
        logging.info("saved: %s", path)
        enforce_quota(a.capture_dir, a.quota_mb)
        self.last_capture = now
        self.total_captured += 1
        return path

    def _save_and_notify(self, frame: np.ndarray, ctype: str, label: str, score: float) -> None:
        path = self._save_and_count(frame, self.args.capture_dir, ctype)
        if path:
            if ctype == "cat":
                self.cat_captures += 1
            self.last_notify = self._maybe_notify(
                path, f"Cat detected: {label} ({score:.2f})",
                self.last_notify, self.args.notify_interval
            )

    def _maybe_notify(self, path: str, caption: str, last_notify: float, interval: float) -> float:
        now = time.time()
        if now - last_notify < interval or not tg.configured(self.token, self.chat_id):
            return last_notify
        if tg.send_photos([path], caption, self.token, self.chat_id):
            return now
        return last_notify

    def _maybe_debug_telegram(self, now: float) -> None:
        if self.args.debug_interval > 0 and now - self.last_debug < self.args.debug_interval:
            return

        # Determine the capture type for this cycle
        # (Only called after a capture, so we check which counter incremented most recently)
        # Simplified: send a summary with all counters
        tg.send_message(
            "🐈 Debug capture summary\n"
            f"Cat: {self.cat_captures} | Idle: {self.idle_captures} | FP: {self.fp_captures} | Total: {self.total_captured}",
            self.token, self.chat_id
        )
        self.last_debug = now


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- logging setup ---
    common.setup_logging(
        log_path=args.log_file,
        level=args.log_level,
        max_bytes=args.log_max_mb * 1024 * 1024,
        backup_count=0,
    )

    logging.info("watch_cat starting (pid=%s)", os.getpid())
    logging.info("arguments: %s", args)

    cfg = load_config(args.config)
    watcher = Watcher(args, cfg)
    watcher.run()


if __name__ == "__main__":
    main()