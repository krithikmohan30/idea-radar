"""History: what has already been proposed, so tomorrow is actually new."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .sources import normalize_url

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "data" / "history.json"
LOG_DIR = ROOT / "ideas"

STOP = {
    "the",
    "a",
    "an",
    "for",
    "and",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "you",
    "app",
    "tool",
    "using",
    "built",
    "build",
    "small",
    "daily",
}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP}


def load() -> list[dict[str, Any]]:
    if not HISTORY.exists():
        return []
    try:
        data = json.loads(HISTORY.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        print("[state] history unreadable, starting fresh")
        return []


def save(records: list[dict[str, Any]]) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(
        json.dumps(records[-400:], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def record(records: list[dict[str, Any]], idea: dict[str, Any], day: str) -> list[dict[str, Any]]:
    records = [r for r in records if r.get("date") != day]
    records.append(
        {
            "date": day,
            "title": idea.get("title", ""),
            "repo_name": idea.get("repo_name", ""),
            "one_liner": idea.get("one_liner", ""),
            "engine": idea.get("engine", ""),
            "fit": (idea.get("profile_fit") or {}).get("score"),
            # Kept so tomorrow can refuse to build on the same article again.
            "sources": [
                str(s.get("url", ""))
                for s in idea.get("sources") or []
                if isinstance(s, dict) and s.get("url")
            ][:6],
        }
    )
    return records


def recent_urls(
    records: list[dict[str, Any]], days: int = 21, today_iso: str | None = None
) -> set[str]:
    """Every URL that backed an idea in the last `days` days.

    Titles drift, links do not. A write-up that stays on a front page for a week
    is one story, and it must not supply the evidence five days running.
    """
    if not records:
        return set()
    anchor = today_iso or today()
    cutoff = (date.fromisoformat(anchor) - timedelta(days=days)).isoformat()
    used: set[str] = set()
    for rec in records:
        if str(rec.get("date", "")) < cutoff:
            continue
        for url in rec.get("sources") or []:
            norm = normalize_url(str(url))
            if norm:
                used.add(norm)
    return used


def similar(a: str, b: str, threshold: float = 0.4) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def duplicates(
    title: str, one_liner: str, records: list[dict[str, Any]], window: int = 60
) -> list[str]:
    hits = []
    for rec in records[-window:]:
        if similar(title, rec.get("title", "")) or similar(one_liner, rec.get("one_liner", "")):
            hits.append(rec.get("title", ""))
    return hits


def append_markdown(idea: dict[str, Any], markdown: str, day: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{day[:7]}.md"
    body = path.read_text(encoding="utf-8") if path.exists() else ""

    # A same-day rerun replaces that day's entry instead of appending a second one.
    entries = ["## " + block.strip() for block in re.split(r"(?m)^## ", body)[1:]]
    entries = [block for block in entries if not block.startswith(f"## {day} ")]
    entries.append(markdown.strip())

    header = f"# Idea log — {day[:7]}\n\nOne idea per day, newest last.\n"
    path.write_text(header + "\n" + "\n\n---\n\n".join(entries) + "\n", encoding="utf-8")
    return path


def today() -> str:
    return date.today().isoformat()


def has_good_idea(records: list[dict[str, Any]], day: str) -> bool:
    """True when today already produced a real (LLM) idea, so a retry can stand down.

    A template fallback does not count: it is worth retrying later in the day when
    the free-tier model is less congested.
    """
    return any(r.get("date") == day and r.get("engine") == "llm" for r in records)
