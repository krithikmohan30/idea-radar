"""Telegram delivery + local preview."""

from __future__ import annotations

import os
import textwrap

import requests

from .sources import USER_AGENT

LIMIT = 3900  # Telegram caps at 4096, leave room for entity overhead


def chunk(text: str, limit: int = LIMIT) -> list[str]:
    """Split on line boundaries so HTML tags are never cut mid-message."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(line) > limit:
            for piece in textwrap.wrap(line, limit):
                chunks.append(piece)
            continue
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks


def send(text: str, token: str | None = None, chat_id: str | None = None) -> int:
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")

    sent = 0
    for part in chunk(text):
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": part,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"telegram {resp.status_code}: {resp.text[:300]}")
        sent += 1
    return sent
