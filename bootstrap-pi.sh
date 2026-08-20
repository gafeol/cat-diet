#!/usr/bin/env bash
# Bootstrap script for cat-diet on a fresh Raspberry Pi (armv7l / aarch64)
# Usage: bash bootstrap-pi.sh

set -euo pipefail

log() { echo -e "\033[1;32m[$(date +'%H:%M:%S')]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

if [[ $EUID -eq 0 ]]; then
    err "Do not run as root. Run as pi/cleitinho user (uses sudo internally)."
    exit 1
fi

log "Updating apt..."
sudo apt update

log "Installing build dependencies for Pillow (no armv7l wheels for Python 3.9+) and python-prctl (picamera2 dep)..."
sudo apt install -y \
    python3-dev \
    build-essential \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libwebp-dev \
    libcap-dev

log "Installing picamera2 (system package is most reliable on Pi OS)..."
sudo apt install -y python3-picamera2

log "Ensuring uv is installed..."
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

log "Creating venv with system site-packages (uses system Python 3.11)..."
rm -rf .venv
rm -f .python-version
uv venv --system-site-packages --python /usr/bin/python3

log "Checking if we need more swap for Pillow compilation (Pi 2B has ~1GB RAM)..."
SWAP_SIZE=$(swapon --show=SIZE --noheadings 2>/dev/null | head -1 | awk '{print $1}' || echo "0")
if [[ "$SWAP_SIZE" -lt 1000000 ]]; then
    log "Low/no swap detected. Adding 1GB swap file..."
    sudo dphys-swapfile swapoff 2>/dev/null || true
    echo "CONF_SWAPSIZE=1024" | sudo tee /etc/dphys-swapfile >/dev/null
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon
fi

log "Syncing Python dependencies with uv..."
cd "$(dirname "$0")"
uv sync --no-build-isolation

log "Installing tflite-runtime into the venv (no deps; numpy comes from system)..."
uv pip install --python .venv/bin/python3 --no-deps "tflite-runtime==2.14.0"

log "Bootstrap complete. Run the watcher with:"
echo "  uv run watch_cat.py --debug --notify"