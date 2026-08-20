# Cat Diet

Smart cat feeder that identifies which cat (Julio or Lina) is at the bowl and automatically switches food plates based on individual dietary needs.

## Goal

Build a system that:

1. **Detects** which cat is at the feeder (Julio or Lina)
2. **Recognizes** the specific cat with a trained model
3. **Actuates** a mechanism to switch food plates automatically
4. **Monitors** feeding behavior and sends stats via Telegram

### Background

- **Julio** and **Lina** are two tabby cats with distinct facial/fur patterns
- Both need different diets (calorie count, food type, portion control)
- Manual feeding is error-prone — the system automates the right food selection

## Hardware

- **Raspberry Pi 2B** + Pi Camera v2 (image capture & inference)
- **Servo motor** or small actuator (switches plates mechanically)
- **2 food plates** loaded with each cat's specific food

## Project Phases

1. **Dataset collection** (`capture_data.py`)
   - Use pretrained MobileNet as a cat-presence detector
   - Capture frames when any cat appears
   - Save images with timestamps for labeling

2. **Training** (`train/` — on MacBook or cloud GPU)
   - Transfer learning from MobileNet v2/v3
   - Fine-tune for 2-class classification: Julio vs Lina
   - Post-training quantization to int8 TFLite for Pi

3. **Custom classifier deployment** (`classify.py`)
   - Load the trained 2-cat model
   - Run inference on Pi (slower on Pi 2B, optimize for ~1-2 FPS)
   - Control servo to switch plates based on prediction

4. **Feeding control** (`feeder.py`)
   - Actuate servo to select left/right plate
   - Dispense portion, monitor via camera
   - Log timestamps, weight, cat identity

## Telegram Integration

- Get notified every 6 hours with collection stats
- Send latest captures on schedule
- Optionally send feeding confirmations

## Files

| File | Purpose |
|------|---------|
| `capture_data.py` | Collects training images using MobileNet cat detection |
| `watch_cat.py` | Long-running, low-energy watcher (change detector + model gate, rolling quota) |
| `classify.py` | (future) Runs custom-trained model for Julio/Lina identification |
| `feeder.py` | (future) Controls servo + acts on classification results |
| `common.py` | Shared helpers (env, labels, cat check, TFLite model loading, state) |
| `telegram.py` | Telegram send primitives (`send_message`, `send_photos`) |
| `quota.py` | Rolling disk quota on the captures directory |
| `mobilenet_v1_1.0_224_quant.tflite` | Detection model (already on device) |
| `labels_mobilenet_quant_v1_224.txt` | Labels for the detection model |
| `.env` | `TG_TOKEN`, `TG_CHAT_ID` for Telegram bot |

## Captured Images

All filenames are prefixed with a day+hour timestamp (`YYYYMMDD_HHMMSS`).
The watcher (`watch_cat.py`) routes saves into three places:

| Folder | Example file | Meaning | Training class |
|--------|--------------|---------|----------------|
| `captures/` | `20260816_143205_cat_jaguar_0.85.jpg` | Motion detected **and** model says a cat is present | `Julio` / `Lina` |
| `captures/false_positives/` | `20260816_143205_fp_dog_0.70.jpg` | Motion triggered, but the model says **not** a cat | Review carefully — mostly `Background`, but a cat here means the detector missed it |
| `captures/bg/` | `20260816_143205_idle.jpg` | Random idle sample (~1/hour) of the empty bowl | `Background` |

- `captures/` photos need to be sorted into `cat1/` (Julio) and `cat2/` (Lina) folders for training.
- `captures/bg/` can be moved to a `nothing/` folder as-is.
- `captures/false_positives/` should be skimmed: legitimate "no cat, just motion" frames go to `nothing/`; frames that *do* contain a cat are detection misses and should be sorted into the cat folders (and flagged — the detector should have caught them).
- For 3-class training the model learns `Julio / Lina / Background`, so the background class gets the empty-view samples.
- Disk is capped by a rolling quota (default 200MB): the oldest files are auto-deleted when it's exceeded.

## Quick Start

1. Set up `.env` with your Telegram bot token and chat ID
2. Put `mobilenet_v1_1.0_224_quant.tflite` and labels in the project folder
3. `uv sync` to install dependencies
4. Run `capture_data.py` to collect training images:
   ```bash
   uv run capture_data.py --runs -1 --interval 5
   ```
5. Or leave `watch_cat.py` running for days (low-energy, event-driven):
   ```bash
   uv run watch_cat.py --notify
   ```
   See the "Captured Images" section for what each folder contains. Tune
   `--motion-threshold` (start ~3.0, raise if it fires on noise, lower if it
   misses real motion) and `--bg-check` (default 30s) to taste.

6. Sort `captures/` into `cat1/` (Julio) and `cat2/` (Lina) folders
7. Train on MacBook/cloud → export `.tflite` → deploy to Pi
8. Build the servo mechanism and write `feeder.py`

## Notes

- The Pi 2B is slow for inference (~200-500ms/frame)
- During idle, `watch_cat.py` runs the full model only every ~30s plus on motion, keeping CPU close to idle the rest of the time
- Use `watch_cat.py --notify` to get a Telegram photo when a cat is detected
- Logs are written to `watch_cat.log` (rotating, max 10 MiB) and can be viewed with `tail -f watch_cat.log` over SSH
