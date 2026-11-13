"""Signal collection.

Every source is independent: one failing API never kills the run. Each returns
a flat list of `Signal` objects that later stages cluster and score.
"""

from __future__ import annotations

import datetime as dt
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from typing import Any

import requests

USER_AGENT = "idea-radar/1.0 (+https://github.com/krithikmohan30/idea-radar)"
TIMEOUT = 20

ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass
class Signal:
    source: str
    title: str
    url: str
    score: int = 0
    comments: int = 0
    tags: list[str] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get(url: str, **kwargs: Any) -> Any:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    headers.update(kwargs.pop("headers", {}))
    resp = requests.get(url, headers=headers, timeout=TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp


def hackernews(limit: int = 25) -> list[Signal]:
    since = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=30)).timestamp())
    url = (
        "https://hn.algolia.com/api/v1/search"
        f"?tags=story&hitsPerPage={limit}"
        f"&numericFilters=points>80,created_at_i>{since}"
    )
    out: list[Signal] = []
    for hit in _get(url).json().get("hits", []):
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        out.append(
            Signal(
                source="hn",
                title=title,
                url=link,
                score=int(hit.get("points") or 0),
                comments=int(hit.get("num_comments") or 0),
                tags=["hn"],
            )
        )
    return out


def github_new_repos(limit: int = 20) -> list[Signal]:
    """Repos created in the last 10 days that are already pulling stars."""
    created = (dt.date.today() - dt.timedelta(days=10)).isoformat()
    url = (
        "https://api.github.com/search/repositories"
        f"?q=created:>{created}+stars:>40&sort=stars&order=desc&per_page={limit}"
    )
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    out: list[Signal] = []
    for repo in _get(url, headers=headers).json().get("items", []):
        out.append(
            Signal(
                source="github",
                title=f"{repo['full_name']}: {repo.get('description') or 'no description'}",
                url=repo["html_url"],
                score=int(repo.get("stargazers_count") or 0),
                tags=[repo.get("language") or "unknown"],
                summary=(repo.get("description") or "")[:280],
            )
        )
    return out


def arxiv(limit: int = 20) -> list[Signal]:
    url = (
        "https://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.SE"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={limit}"
    )
    root = ET.fromstring(_get(url, headers={"Accept": "application/atom+xml"}).content)
    out: list[Signal] = []
    for entry in root.findall(f"{ATOM}entry"):
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        link = entry.findtext(f"{ATOM}id") or ""
        if not title or not link:
            continue
        summary = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
        out.append(
            Signal(
                source="arxiv",
                title=title,
                url=link,
                tags=["paper"],
                summary=summary[:400],
            )
        )
    return out


def lobsters(limit: int = 25) -> list[Signal]:
    out: list[Signal] = []
    for story in _get("https://lobste.rs/hottest.json").json()[:limit]:
        out.append(
            Signal(
                source="lobsters",
                title=story.get("title") or "",
                url=story.get("url") or story.get("comments_url", ""),
                score=int(story.get("score") or 0),
                comments=int(story.get("comment_count") or 0),
                tags=story.get("tags") or [],
            )
        )
    return [s for s in out if s.title]


def devto(limit: int = 20) -> list[Signal]:
    out: list[Signal] = []
    for art in _get(f"https://dev.to/api/articles?top=7&per_page={limit}").json():
        out.append(
            Signal(
                source="devto",
                title=art.get("title") or "",
                url=art.get("url") or "",
                score=int(art.get("positive_reactions_count") or 0),
                comments=int(art.get("comments_count") or 0),
                tags=art.get("tag_list") or [],
            )
        )
    return [s for s in out if s.title]


def show_hn(limit: int = 20) -> list[Signal]:
    """Show HN = people shipping. Best raw material for 'what's missing'."""
    url = f"https://hn.algolia.com/api/v1/search_by_date?tags=show_hn&hitsPerPage={limit}"
    out: list[Signal] = []
    for hit in _get(url).json().get("hits", []):
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        out.append(
            Signal(
                source="show_hn",
                title=title,
                url=f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                score=int(hit.get("points") or 0),
                comments=int(hit.get("num_comments") or 0),
                tags=["show_hn"],
            )
        )
    return out


SOURCES = {
    "hn": hackernews,
    "show_hn": show_hn,
    "github": github_new_repos,
    "arxiv": arxiv,
    "lobsters": lobsters,
    "devto": devto,
}


def collect(only: Iterable[str] | None = None) -> tuple[list[Signal], list[str]]:
    """Fetch all sources. Returns (signals, errors) so failures stay visible."""
    names = list(only) if only else list(SOURCES)
    signals: list[Signal] = []
    errors: list[str] = []
    for name in names:
        fn = SOURCES.get(name)
        if fn is None:
            errors.append(f"{name}: unknown source")
            continue
        try:
            items = fn()
            signals.extend(items)
            print(f"[sources] {name}: {len(items)} items")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"[sources] {name}: FAILED ({type(exc).__name__})")
    return signals, errors


def top_signals(signals: list[Signal], limit: int = 40, per_source: int = 8) -> list[Signal]:
    """Interleave sources instead of letting one loud feed own the prompt.

    Without this, a single big news story on HN supplies every candidate and the
    ideas turn into news commentary instead of buildable tools.
    """
    by_source: dict[str, list[Signal]] = {}
    for sig in signals:
        by_source.setdefault(sig.source, []).append(sig)

    queues = [
        sorted(items, key=lambda s: s.score + 2 * s.comments, reverse=True)[:per_source]
        for items in by_source.values()
    ]

    out: list[Signal] = []
    seen: set[str] = set()
    for row in zip_longest(*queues):
        for sig in row:
            if sig is None:
                continue
            key = sig.title.lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            out.append(sig)
            if len(out) >= limit:
                return out
    return out
