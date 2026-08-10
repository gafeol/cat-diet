#!/usr/bin/env python3
"""Pi Camera + int8 TFLite: detect any cat in frame and save raw frames for dataset collection.
Sends Telegram updates every 6 hours with photo stats and latest captures."""
import argparse
import glob
import os
import time
from datetime import datetime, timezone

import numpy as np
import requests
from PIL import Image
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

MODEL = "mobilenet_v1_1.0_224_quant.tflite"
LABELS = "labels_mobilenet_quant_v1_224.txt"
RES = (1280, 720)
INPUT_SIZE = 224
SAVE_DIR = "captures"

CAT_KEYWORDS = ("cat", "tabby", "lynx", "cougar", "jaguar", "leopard", "cheetah")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env(path):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env(os.path.join(_SCRIPT_DIR, ".env"))
load_env(os.path.join(os.path.dirname(_SCRIPT_DIR), ".env"))

BOT_TOKEN = os.environ.get("TG_TOKEN", "PASTE_YOUR_BOT_TOKEN")
CHAT_ID = os.environ.get("TG_CHAT_ID", "PASTE_YOUR_CHAT_ID")
TELEGRAM_PHOTO_URL = "https://api.telegram.org/bot{token}/sendPhoto"
TELEGRAM_MSG_URL = "https://api.telegram.org/bot{token}/sendMessage"

STATE_FILE = os.path.join(SAVE_DIR, ".state.json")


def load_labels(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


def is_cat(label):
    low = label.lower()
    return any(k in low for k in CAT_KEYWORDS)


def send_telegram_message(text):
    try:
        requests.post(
            TELEGRAM_MSG_URL.format(token=BOT_TOKEN),
            data={"chat_id": CHAT_ID, "text": text},
            timeout=30,
        )
    except Exception as e:
        print(f"  -> Telegram message failed: {e}")


def send_telegram_photos(photo_paths, caption):
    ok_all = True
    for path in photo_paths:
        try:
            with open(path, "rb") as f:
                r = requests.post(
                    TELEGRAM_PHOTO_URL.format(token=BOT_TOKEN),
                    data={"chat_id": CHAT_ID, "caption": caption},
                    files={"photo": f},
                    timeout=30,
                )
            if not r.ok:
                ok_all = False
        except Exception as e:
            print(f"  -> Telegram photo failed: {e}")
            ok_all = False
    return ok_all


def get_hour_bucket():
    """Return current hour bucket (0, 6, 12, 18) based on UTC time."""
    now_utc = datetime.now(timezone.utc)
    return (now_utc.hour // 6) * 6


def send_periodic_update(total_captured, last_bucket):
    """Send a Telegram update every 6 hours with stats and latest 3 photos."""
    current_bucket = get_hour_bucket()
    if current_bucket == last_bucket:
        return last_bucket

    files = sorted(glob.glob(os.path.join(SAVE_DIR, "*.jpg")), key=os.path.getmtime)
    latest = files[-3:] if len(files) >= 3 else files

    text_lines = [f"📊 Dataset collection update",
                  f"Total photos captured so far: {total_captured}",
                  f"Photos in folder: {len(files)}"]
    if latest:
        text_lines.append(f"Latest {len(latest)} captures:")
        for p in latest:
            text_lines.append(f"  📷 {os.path.basename(p)}")
    else:
        text_lines.append("No photos captured yet.")

    send_telegram_message("\n".join(text_lines))

    if latest:
        send_telegram_photos(latest, f"Latest capture ({len(latest)} shown)")

    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    print(f"  -> 6-hour update sent at {now_str} (bucket {current_bucket})")
    return current_bucket


def load_state():
    """Load persistent state (total_captured, last_bucket) from file."""
    if os.path.exists(STATE_FILE):
        try:
            import json
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"total_captured": 0, "last_bucket": None}


def save_state(state):
    """Save persistent state to file."""
    import json
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"  -> Failed to save state: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=-1, help="frames to classify, -1 = forever")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--interval", type=float, default=5.0,
                        help="min seconds between captures (avoid saving too many similar frames)")
    args = parser.parse_args()

    labels = load_labels(LABELS)

    interpreter = Interpreter(model_path=MODEL, num_threads=1)
    interpreter.allocate_tensors()
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    print(f"input : {in_d['shape']} dtype={in_d['dtype']} quant={in_d['quantization']}")
    print(f"output: {out_d['shape']} dtype={out_d['dtype']} quant={out_d['quantization']}")

    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": RES, "format": "RGB888"}))
    cam.start()
    time.sleep(2)

    os.makedirs(SAVE_DIR, exist_ok=True)
    state = load_state()
    total_captured = state.get("total_captured", 0)
    last_bucket = state.get("last_bucket", None)
    last_capture = 0.0
    run = 0
    while args.runs < 0 or run < args.runs:
        frame = cam.capture_array("main")
        img = Image.fromarray(frame)
        arr = np.expand_dims(
            np.array(img.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS), dtype=np.uint8), axis=0)

        t0 = time.perf_counter()
        interpreter.set_tensor(in_d["index"], arr)
        interpreter.invoke()
        out = interpreter.get_tensor(out_d["index"])[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        top = int(np.argmax(out))
        score = out[top] / 255.0 if out.dtype == np.uint8 else out[top]
        label = labels[top] if top < len(labels) else f"class_{top}"
        print(f"run {run}: {elapsed_ms:6.0f} ms  -> {label} ({score:.2f})")

        if is_cat(label) and score >= args.threshold:
            if time.time() - last_capture >= args.interval:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SAVE_DIR, f"{ts}_{label}_{score:.2f}.jpg")
                img.save(path, quality=85)
                print(f"  -> saved: {path}")
                last_capture = time.time()
                total_captured += 1
                save_state({"total_captured": total_captured, "last_bucket": last_bucket})

        last_bucket = send_periodic_update(total_captured, last_bucket)
        save_state({"total_captured": total_captured, "last_bucket": last_bucket})

        run += 1

    cam.stop()


if __name__ == "__main__":
    main()
