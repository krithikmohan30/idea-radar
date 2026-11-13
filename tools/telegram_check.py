"""Diagnose Telegram wiring without leaking the token or the chat id.

Prints only what is safe in a public Actions log: the bot username, how many
updates are waiting, and a masked view of each chat id.
"""

from __future__ import annotations

import os
import sys

import requests


def mask(value: str) -> str:
    value = value.strip()
    return "***" + value[-4:] if len(value) > 4 else "***"


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token:
        print("TELEGRAM_BOT_TOKEN is empty")
        return 1

    me = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=20).json()
    if not me.get("ok"):
        print(f"token rejected: {me.get('description')}")
        return 1
    print(f"bot: @{me['result']['username']} (id {me['result']['id']})")

    if not chat_id:
        print("TELEGRAM_CHAT_ID is empty")
        return 1
    if chat_id != chat_id.strip() or chat_id.startswith(("'", '"')):
        print("warning: chat id has stray whitespace or quotes")
    print(f"configured chat id: {mask(chat_id)} (len {len(chat_id)})")

    chat = requests.get(
        f"https://api.telegram.org/bot{token}/getChat",
        params={"chat_id": chat_id},
        timeout=20,
    ).json()
    if chat.get("ok"):
        print(
            f"getChat ok: type={chat['result'].get('type')} title={chat['result'].get('title', '-')}"
        )
        return 0
    print(f"getChat failed: {chat.get('description')}")

    updates = requests.get(
        f"https://api.telegram.org/bot{token}/getUpdates",
        params={"limit": 20},
        timeout=20,
    ).json()
    found = []
    for update in updates.get("result", []):
        for key in ("message", "edited_message", "channel_post", "my_chat_member"):
            chat_obj = (update.get(key) or {}).get("chat")
            if chat_obj:
                found.append((chat_obj.get("type"), str(chat_obj.get("id"))))
    print(f"pending updates: {len(found)}")
    for kind, found_id in found:
        match = "matches configured" if found_id == chat_id else "differs"
        print(f"  chat type={kind} id={mask(found_id)} {match}")

    if not found:
        print("open Telegram, send any message to the bot, then re-run")
    return 1


if __name__ == "__main__":
    sys.exit(main())
