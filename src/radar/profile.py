"""The candidate's own GitHub footprint, used to stop the bot suggesting
things that already exist in the portfolio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from .sources import USER_AGENT

DEFAULT_USER = "krithikmohan30"
CACHE = Path(__file__).resolve().parents[2] / "data" / "profile.json"


def _api(url: str) -> Any:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_profile(user: str | None = None) -> dict[str, Any]:
    """Live profile. Falls back to the committed cache, then to an empty shell."""
    user = user or os.environ.get("PROFILE_USER") or os.environ.get("GITHUB_USER") or DEFAULT_USER
    try:
        repos = _api(f"https://api.github.com/users/{user}/repos?per_page=100&sort=pushed")
        mine = [
            {
                "name": r["name"],
                "description": r.get("description") or "",
                "language": r.get("language") or "",
                "stars": r.get("stargazers_count", 0),
                "topics": r.get("topics") or [],
            }
            for r in repos
            if not r.get("fork")
        ]
        readme = ""
        try:
            resp = requests.get(
                f"https://raw.githubusercontent.com/{user}/{user}/HEAD/README.md",
                headers={"User-Agent": USER_AGENT},
                timeout=20,
            )
            if resp.ok:
                readme = resp.text[:4000]
        except requests.RequestException:
            pass
        profile = {"user": user, "repos": mine, "readme": readme}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        return profile
    except Exception as exc:
        print(f"[profile] live fetch failed ({exc}); trying cache")
        if CACHE.exists():
            return json.loads(CACHE.read_text(encoding="utf-8"))
        return {"user": user, "repos": [], "readme": ""}


def summary(profile: dict[str, Any]) -> str:
    """Compact text block for the prompt."""
    lines = [f"GitHub user: {profile.get('user', '?')}"]
    if profile.get("repos"):
        lines.append("Existing public repos (do NOT propose these again):")
        for repo in profile["repos"][:40]:
            desc = (repo.get("description") or "").strip()[:110]
            lines.append(f"- {repo['name']} [{repo.get('language') or 'n/a'}] {desc}")
    if profile.get("readme"):
        lines.append("Profile README excerpt:")
        lines.append(profile["readme"][:1800])
    return "\n".join(lines)
