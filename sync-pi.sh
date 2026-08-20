#!/usr/bin/env bash
# Sync project files to the Pi, excluding heavy/local dirs (venvs, captures, model).
# Usage: PI_HOST=cleitinho@pi:~ bash sync-pi.sh
#        (or set PI_HOST in your shell config; defaults to cleitinho@pi)

set -euo pipefail

PI_HOST="${PI_HOST:-cleitinho@192.168.15.5}"

rsync -av \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='captures' \
    --exclude='data' \
    --exclude='*.log' \
    --exclude='.state.json' \
    "$(dirname "$0")/" \
    "${PI_HOST}:~/cat-diet/"

echo "Synced. On the Pi:"
echo "  uv sync --no-build-isolation"
echo "  uv run watch_cat.py --notify"
