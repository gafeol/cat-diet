#!/usr/bin/env python3
"""Minimal Telegram notifications for cat-diet scripts with logging support."""

from __future__ import annotations

import logging
import os

import requests

DEFAULT_TOKEN = "PASTE_YOUR_BOT_TOKEN"
DEFAULT_CHAT_ID = "PASTE_YOUR_CHAT_ID"

_BASE_URL = "https://api.telegram.org/bot{token}/{endpoint}"

logger = logging.getLogger(__name__)


def _post(endpoint: str, token: str, **kwargs) -> bool:
    """Generic POST to Telegram Bot API."""
    try:
        r = requests.post(
            _BASE_URL.format(token=token, endpoint=endpoint),
            timeout=30,
            **kwargs,
        )
        return bool(r.ok)
    except Exception as e:
        logger.warning("Telegram %s failed: %s", endpoint, e)
        return False


def configured(token: str, chat_id: str) -> bool:
    """True when a real token/chat_id are available (placeholders return False)."""
    return (
        bool(token)
        and token != DEFAULT_TOKEN
        and bool(chat_id)
        and chat_id != DEFAULT_CHAT_ID
    )


def send_message(text: str, token: str, chat_id: str, timeout: int = 30) -> bool:
    return _post("sendMessage", token, data={"chat_id": chat_id, "text": text})


def get_updates(token: str, offset: int = 0, timeout: int = 30) -> list[dict]:
    """Long-poll for incoming updates. Returns raw update dicts (may be empty)."""
    try:
        r = requests.get(
            _BASE_URL.format(token=token, endpoint="getUpdates"),
            params={"offset": offset, "timeout": timeout},
            timeout=timeout + 10,
        )
        r.raise_for_status()
        return r.json().get("result", [])
    except Exception as e:
        logger.warning("Telegram getUpdates failed: %s", e)
        return []


def send_photos(
    photo_paths: list[str], caption: str, token: str, chat_id: str, timeout: int = 30
) -> bool:
    ok_all = True
    for path in photo_paths:
        try:
            with open(path, "rb") as f:
                ok = _post(
                    "sendPhoto",
                    token,
                    data={"chat_id": chat_id, "caption": caption},
                    files={"photo": f},
                )
            ok_all = ok and ok_all
        except Exception as e:
            logger.warning("Telegram photo failed: %s", e)
            ok_all = False
    return ok_all


def env_token() -> str:
    return os.environ.get("TG_TOKEN", DEFAULT_TOKEN)


def env_chat_id() -> str:
    return os.environ.get("TG_CHAT_ID", DEFAULT_CHAT_ID)