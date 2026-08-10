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
| `classify.py` | (future) Runs custom-trained model for Julio/Lina identification |
| `feeder.py` | (future) Controls servo + acts on classification results |
| `mobilenet_v1_1.0_224_quant.tflite` | Detection model (already on device) |
| `labels_mobilenet_quant_v1_224.txt` | Labels for the detection model |
| `.env` | `TG_TOKEN`, `TG_CHAT_ID` for Telegram bot |

## Quick Start

1. Set up `.env` with your Telegram bot token and chat ID
2. Put `mobilenet_v1_1.0_224_quant.tflite` and labels in the project folder
3. Run `capture_data.py` to collect training images:
   ```bash
   pip install numpy pillow requests tflite-runtime picamera2
   python capture_data.py --runs -1 --interval 5
   ```
4. Sort `captures/` into `cat1/` (Julio) and `cat2/` (Lina) folders
5. Train on MacBook/cloud → export `.tflite` → deploy to Pi
6. Build the servo mechanism and write `feeder.py`

## Notes

- The Pi 2B is slow for inference (~200-500ms/frame) — optimize by skipping every other frame
- Use the 6-hour Telegram updates to track dataset collection progress
