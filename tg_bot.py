#!/usr/bin/env python3
"""Simple long-polling Telegram bot for cat-diet.

Listens for commands in the authorized chat (TG_CHAT_ID from .env) and answers
with photos / stats from the capture directory. Uses Bot API getUpdates long
polling, so no public webhook URL is required (works behind NAT on the Pi).

Commands:
  /help, /start, help   show this help
  /last [n]             send the last n cat photos (default 1, max 10)
  /stats, stats         send capture counters
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import time

import common
import telegram as tg

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HELP_TEXT = (
    "🐈 cat-diet bot\n"
    "/last [n] — last n cat photos (default 1, max 10)\n"
    "/stats    — capture counts\n"
    "/help     — this message"
)

MAX_PHOTOS = 10


def _is_cat_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(k in name for k in common.CAT_KEYWORDS) or "_cat" in name


def find_last_cat_photos(capture_dir: str, n: int = 1) -> list[str]:
    """Newest *n* JPEGs in *capture_dir* whose filename hints at a cat (newest first)."""
    candidates = [
        path
        for path in glob.glob(os.path.join(capture_dir, "*.jpg"))
        if _is_cat_file(path)
    ]
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[: max(0, min(n, MAX_PHOTOS))]


def stats_text(capture_dir: str) -> str:
    state = common.load_json(os.path.join(capture_dir, ".state.json"), {})
    total = state.get("total_captured", 0)
    cats = len(glob.glob(os.path.join(capture_dir, "*_cat*.jpg")))
    jpgs = len(glob.glob(os.path.join(capture_dir, "**", "*.jpg"), recursive=True))
    return (
        f"Captures total: {total}\n"
        f"Cat photos: {cats}\n"
        f"All JPEGs on disk: {jpgs}"
    )


def handle_message(text: str, capture_dir: str) -> tuple[str, list[str]]:
    """Return (reply, photo_paths) for a given command text."""
    low = (text or "").strip().lower()
    if low in ("/start", "/help", "help", "commands"):
        return HELP_TEXT, []
    if low in ("/stats", "stats", "statistics", "counts"):
        return stats_text(capture_dir), []
    if low.startswith("/last") or any(
        low.startswith(w) for w in ("last ", "latest", "last photo", "last picture", "last cat", "photo", "picture", "foto", "cat pic")
    ):
        n = 1
        for word in low.split():
            if word.isdigit():
                n = int(word)
                break
        n = max(1, min(n, MAX_PHOTOS))
        photos = find_last_cat_photos(capture_dir, n)
        if not photos:
            return "No cat pictures yet.", []
        count = len(photos)
        return f"Latest {count} cat photo{'s' if count > 1 else ''}:", photos
    return (
        "Unknown command. Try /last [n], /stats or /help.",
        [],
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", default="captures", help="capture directory")
    p.add_argument("--poll", type=int, default=30, help="long-poll timeout (s)")
    args = p.parse_args(argv)

    common.load_env(os.path.join(_SCRIPT_DIR, ".env"))
    common.load_env(os.path.join(os.path.dirname(_SCRIPT_DIR), ".env"))
    token = tg.env_token()
    chat_id = tg.env_chat_id()

    common.setup_logging(log_path="tg_bot.log", level="INFO")
    logging.info("tg_bot starting (pid=%s)", os.getpid())
    if not tg.configured(token, chat_id):
        logging.error("Telegram not configured; set TG_TOKEN/TG_CHAT_ID in .env")
        return

    offset = 0
    while True:
        for update in tg.get_updates(token, offset=offset, timeout=args.poll):
            offset = update["update_id"] + 1
            msg = update.get("message", {})
            if str(msg.get("chat", {}).get("id")) != str(chat_id):
                logging.info("ignoring update from chat %s", msg.get("chat", {}).get("id"))
                continue
            text = msg.get("text") or ""
            logging.info("command from %s: %r", msg.get("from", {}).get("username"), text)

            reply, photos = handle_message(text, args.capture_dir)
            if photos:
                tg.send_photos(photos, reply, token, chat_id)
            else:
                tg.send_message(reply, token, chat_id)
        time.sleep(0.5)


if __name__ == "__main__":
    main()