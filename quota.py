#!/usr/bin/env python3
"""Rolling disk quota for a directory tree of JPEG captures with type hints."""

from __future__ import annotations

import logging
import os

MB = 1024 * 1024


def _jpg_paths(directory: str) -> list[str]:
    """Return all .jpg file paths under directory (recursive)."""
    return [
        os.path.join(root, f)
        for root, _dirs, files in os.walk(directory)
        for f in files
        if f.lower().endswith(".jpg")
    ]


def count_jpgs(directory: str) -> int:
    return len(_jpg_paths(directory))


def enforce_quota(directory: str, quota_mb: float) -> int:
    """Delete oldest *.jpg files until the directory is <= quota_mb. Returns count deleted."""
    limit: float = float(quota_mb) * MB
    jpgs: list[str] = sorted(_jpg_paths(directory), key=os.path.getmtime)
    if not jpgs:
        return 0

    total: int = 0
    for p in jpgs:
        try:
            total += os.path.getsize(p)
        except OSError:
            pass

    deleted: int = 0
    while total > limit and jpgs:
        oldest: str = jpgs.pop(0)
        try:
            size: int = os.path.getsize(oldest)
        except OSError:
            size: int = 0
        try:
            os.remove(oldest)
            total -= size
            deleted += 1
        except OSError:
            pass
    if deleted:
        logger = logging.getLogger(__name__)
        logger.info(
            "quota: removed %d oldest file(s) to stay under %.1f MiB", deleted, quota_mb
        )
    return deleted