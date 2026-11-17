"""Idea schema, validation, and rendering (Telegram HTML + repo Markdown)."""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

REQUIRED = ("title", "one_liner", "why_now", "build_plan")


@dataclass
class Idea:
    title: str
    one_liner: str
    why_now: str
    build_plan: dict[str, str] = field(default_factory=dict)
    stack: list[str] = field(default_factory=list)
    repo_name: str = ""
    readme_hook: str = ""
    profile_fit: dict[str, Any] = field(default_factory=dict)
    risk: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)
    # The candidates this idea beat, so the narrowing is visible and auditable.
    shortlist: list[dict[str, Any]] = field(default_factory=list)
    engine: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)[:40] or "idea"


def _str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [p.strip() for p in re.split(r"[,\n]", value) if p.strip()]
    return []


def _plan(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if v}
    if isinstance(value, list):
        return {f"Day {i + 1}": str(v) for i, v in enumerate(value) if v}
    if isinstance(value, str) and value.strip():
        return {"Plan": value.strip()}
    return {}


def _shortlist(value: Any) -> list[dict[str, Any]]:
    """The ranked candidates the pick beat. Tolerant of sloppy model shapes."""
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value, 1):
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "").strip()
            if not title:
                continue
            out.append(
                {
                    "rank": item.get("rank") or index,
                    "title": title,
                    "source": str(item.get("source") or "").strip(),
                    "score": item.get("score"),
                    "why": str(item.get("why") or item.get("reason") or "").strip(),
                }
            )
    return out[:6]


WRAPPER_KEYS = (
    "pick",
    "idea",
    "project",
    "selection",
    "selected",
    "winner",
    "recommendation",
    "result",
    "data",
)


def _unwrap(payload: dict[str, Any]) -> dict[str, Any]:
    """Find the idea inside whatever wrapper key the model chose.

    The prompt asks for {"shortlist": [...], "pick": {...}}, but models rename that
    key freely. Guessing wrong turns a good response into "missing fields: title".
    """
    if payload.get("title"):
        return payload
    for key in WRAPPER_KEYS:
        value = payload.get(key)
        if isinstance(value, dict) and (value.get("title") or value.get("one_liner")):
            return {**payload, **value}
    for value in payload.values():
        if isinstance(value, dict) and value.get("title"):
            return {**payload, **value}
    return payload


def from_json(payload: dict[str, Any], engine: str = "llm") -> Idea:
    """Tolerant normaliser: the model is allowed to be slightly sloppy."""
    payload = _unwrap(payload) if isinstance(payload, dict) else {}

    fit = payload.get("profile_fit") or {}
    if not isinstance(fit, dict):
        fit = {"score": fit}

    idea = Idea(
        title=str(payload.get("title") or "").strip(),
        one_liner=str(payload.get("one_liner") or payload.get("pitch") or "").strip(),
        why_now=str(payload.get("why_now") or payload.get("why") or "").strip(),
        build_plan=_plan(payload.get("build_plan")),
        stack=_str_list(payload.get("stack")),
        repo_name=str(payload.get("repo_name") or "").strip(),
        readme_hook=str(payload.get("readme_hook") or "").strip(),
        profile_fit=fit,
        risk=str(payload.get("risk") or "").strip(),
        sources=[
            {"title": str(s.get("title", "")), "url": str(s.get("url", ""))}
            for s in (payload.get("sources") or [])
            if isinstance(s, dict)
        ],
        shortlist=_shortlist(payload.get("shortlist")),
        engine=engine,
    )
    if not idea.repo_name:
        idea.repo_name = slug(idea.title)
    if not idea.stack:
        idea.stack = ["Python"]

    missing = [k for k in REQUIRED if not getattr(idea, k)]
    if missing:
        raise ValueError(f"idea missing fields: {', '.join(missing)}")
    return idea


def fit_score(idea: Idea) -> int:
    raw = idea.profile_fit.get("score", 0)
    try:
        return max(0, min(10, int(float(raw))))
    except (TypeError, ValueError):
        return 0


def calibrate(idea: Idea) -> Idea:
    """Stop the scorer flattering everything.

    Four consecutive ideas came back 9, 9, 9, 8, which is not a judgement, it is a
    habit. Two rules the model cannot argue with:

    - 10 is reserved. A student portfolio does not warrant a perfect score, so a 10
      becomes a 9 and says so.
    - If every shortlisted candidate scored 8 or better, nothing was actually
      ranked, so the winner is capped at 8.
    """
    score = fit_score(idea)
    note = ""

    scores: list[int] = []
    for item in idea.shortlist:
        try:
            scores.append(int(float(item.get("score"))))
        except (TypeError, ValueError):
            continue

    if score > 9:
        score = 9
        note = "capped from 10: nothing in a student portfolio is a certain 10"
    elif len(scores) >= 3 and min(scores) >= 8 and score > 8:
        score = 8
        note = "capped at 8: every shortlisted candidate scored 8+, so none was really ranked"

    if note:
        fit = dict(idea.profile_fit)
        reason = str(fit.get("reason", "")).strip()
        fit["score"] = score
        fit["reason"] = f"{reason} ({note})".strip()
        idea.profile_fit = fit
    return idea


def render_telegram(idea: Idea, date: str, errors: list[str] | None = None) -> str:
    e = html.escape
    plan = "\n".join(f"  <b>{e(str(k))}</b> {e(str(v))}" for k, v in idea.build_plan.items())
    stack = " · ".join(e(s) for s in idea.stack)
    sources = "\n".join(
        f'  <a href="{e(s["url"])}">{e(s["title"][:70])}</a>' for s in idea.sources[:3]
    )
    fit_reason = e(str(idea.profile_fit.get("reason", "")).strip())

    parts = [
        f"<b>Idea Radar</b> · {e(date)}",
        "",
        f"<b>{e(idea.title)}</b>",
        e(idea.one_liner),
        "",
        f"<b>Why now</b>\n{e(idea.why_now)}",
    ]
    if plan:
        parts += ["", f"<b>2-day build</b>\n{plan}"]
    parts += [
        "",
        f"<b>Stack</b> {stack}",
        f"<b>Repo name</b> <code>{e(idea.repo_name)}</code>",
    ]
    if idea.readme_hook:
        parts += ["", f"<b>README hook</b>\n<i>{e(idea.readme_hook)}</i>"]
    if fit_reason or idea.profile_fit:
        parts += ["", f"<b>Profile fit</b> {fit_score(idea)}/10 — {fit_reason}"]
    if idea.risk:
        parts += [f"<b>Risk</b> {e(idea.risk)}"]
    ranked = [
        f"{e(str(item.get('title', ''))[:48])} ({item.get('score', '?')})"
        for item in idea.shortlist[1:4]
        if item.get("title")
    ]
    if ranked:
        parts += ["", "<b>Also considered</b> " + " · ".join(ranked)]
    if sources:
        parts += ["", "<b>Signal</b>", sources]
    if idea.engine != "llm":
        parts += [
            "",
            "<i>Fallback engine — no API key or the model failed. Treat this as a lead, not a spec.</i>",
        ]
    if errors:
        parts += ["", f"<i>{len(errors)} source(s) failed today.</i>"]
    return "\n".join(parts).strip()


def render_markdown(idea: Idea, date: str) -> str:
    plan = "\n".join(f"- **{k}** — {v}" for k, v in idea.build_plan.items())
    sources = "\n".join(f"- [{s['title']}]({s['url']})" for s in idea.sources[:5])
    ranked = "\n".join(
        f"{i}. **{item['title']}** — {item.get('source') or 'signal'}, "
        f"{item.get('score', '?')}/10" + (f" — {item['why']}" if item.get("why") else "")
        for i, item in enumerate(idea.shortlist[:5], 1)
    )
    return (
        f"## {date} — {idea.title}\n\n"
        f"> {idea.one_liner}\n\n"
        f"**Why now:** {idea.why_now}\n\n"
        f"**Profile fit:** {fit_score(idea)}/10 — {idea.profile_fit.get('reason', '')}\n\n"
        f"**Stack:** {', '.join(idea.stack)}\n\n"
        f"**Repo:** `{idea.repo_name}`\n\n"
        f"{plan}\n\n"
        f"**README hook:** {idea.readme_hook}\n\n"
        f"{('**Ranked:**' + chr(10) + ranked + chr(10) + chr(10)) if ranked else ''}"
        f"{('**Risk:** ' + idea.risk + chr(10) + chr(10)) if idea.risk else ''}"
        f"{('**Signal:**' + chr(10) + sources) if sources else ''}\n"
    )


def load_json(text: str) -> dict[str, Any]:
    """Models love wrapping JSON in prose or fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])
