"""LLM providers: Gemini (free tier), Groq (free tier), and a deterministic
fallback so a missing key or a 429 never produces an empty morning.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests

from .idea import Idea, from_json, load_json, slug
from .sources import Signal

SYSTEM = """You are a senior engineer who mentors a student developer on portfolio strategy.
Your only job: turn today's technology signal into ONE small, demo-able project that
would make a recruiter stop scrolling.

Hard rules:
- Shippable by one person in two evenings (~6-8 focused hours).
- Must produce something visible in under 30 seconds: a live link, a CLI demo, a chart, a before/after.
- Must not duplicate anything already in the user's repos.
- Must be specific. "An AI chat app" is rejected. "A CLI that diffs two OpenAPI specs and flags breaking changes with severity" is accepted.
- Reject: todo apps, clones, tutorial rehashes, thin GPT wrappers, anything needing paid infra or a team.
- Prefer: a sharp wedge of a current trend, a developer tool, a thing that is funny or useful enough to share.
- Cite 2-4 of the provided signals as evidence, using their exact URLs.
Return JSON only. No prose, no markdown fences."""

SCHEMA_HINT = """{
  "title": "short product name + 4-8 word descriptor",
  "one_liner": "one sentence, what it does for whom",
  "why_now": "2-3 sentences tying it to today's signals",
  "build_plan": {"Day 1": "...", "Day 2": "..."},
  "stack": ["3-5 concrete tools"],
  "repo_name": "kebab-case repo name",
  "readme_hook": "the first line of the README that makes someone star it",
  "profile_fit": {"score": 8, "reason": "why this fills a gap in THIS profile, or what it proves"},
  "risk": "the one thing most likely to go wrong",
  "sources": [{"title": "...", "url": "https://exact-url-from-signals"}]
}"""


def build_prompt(profile_text: str, signals: list[Signal], avoid: list[str]) -> str:
    signal_lines = []
    for i, s in enumerate(signals, 1):
        tags = ",".join(s.tags[:4])
        signal_lines.append(
            f"{i}. [{s.source}{'/' + tags if tags else ''}] {s.title}\n"
            f"   score={s.score} comments={s.comments} {s.url}"
        )
    avoid_block = "\n".join(f"- {t}" for t in avoid[:30]) or "- (none yet)"
    return f"""## The developer
{profile_text}

## Today's signal ({len(signals)} items, freshest and loudest first)
{chr(10).join(signal_lines)}

## Already proposed (do not repeat, do not paraphrase these)
{avoid_block}

## Task
Propose exactly one project. Schema:
{SCHEMA_HINT}

Constraints: two evenings max, visible demo in 30 seconds, no duplication of the repos above.
Be concrete about what gets built on day one versus day two. Score profile_fit honestly (1-10);
a 10 means it would genuinely change how a recruiter reads this profile."""


class LLMError(RuntimeError):
    pass


def _post(
    url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json", **(headers or {})},
                timeout=90,
            )
        except requests.RequestException as exc:
            last = exc
        else:
            if resp.ok:
                return resp.json()
            last = LLMError(f"{resp.status_code}: {resp.text[:400]}")
            if resp.status_code not in {429, 500, 502, 503, 504}:
                raise last
        if attempt < 2:
            time.sleep(2 + attempt * 4)
    raise LLMError(str(last))


def gemini(prompt: str, api_key: str, model: str | None = None) -> dict[str, Any]:
    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 1.0,
            "maxOutputTokens": 4096,
        },
    }
    data = _post(url, body)
    try:
        return load_json(data["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected Gemini payload: {json.dumps(data)[:300]}") from exc


def groq(prompt: str, api_key: str, model: str | None = None) -> dict[str, Any]:
    model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    body = {
        "model": model,
        "temperature": 0.9,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    data = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        body,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        return load_json(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected Groq payload: {json.dumps(data)[:300]}") from exc


def template(prompt_signals: list[Signal]) -> dict[str, Any]:
    """No-key fallback: a real lead built from the most buildable signal.

    Ship-shaped sources (what people are shipping, what repos are taking off) beat
    a news story when there is no model to do the reasoning.
    """
    if not prompt_signals:
        raise LLMError("no signals to build a fallback from")
    buildable = ("show_hn", "github", "devto", "lobsters")
    pool = [s for s in prompt_signals if s.source in buildable] or prompt_signals
    top = max(pool, key=lambda s: s.score + 2 * s.comments)
    clean = re.sub(r"^(show|ask|tell) hn:\s*", "", top.title, flags=re.I).strip(" .")
    words = [w.strip(".,:;()[]") for w in clean.split() if len(w) > 3][:5]
    topic = " ".join(words) or "today's top signal"
    return {
        "title": f"Signal Sifter: a daily triage tool for {topic}",
        "one_liner": (
            f"A small CLI/site that tracks '{top.title[:70]}' and turns new mentions into a "
            "ranked digest you can act on."
        ),
        "why_now": f"This is today's loudest item ({top.score} points). The angle is not the story itself, "
        "but the missing tooling around it: nobody has packaged the follow-up reads, alternatives, and "
        "open questions into one page.",
        "build_plan": {
            "Day 1": "Fetch the source, normalise items into SQLite, rank by engagement.",
            "Day 2": "Generate a one-page HTML digest, deploy to Vercel/Pages, write the README.",
        },
        "stack": ["Python", "requests", "SQLite", "GitHub Pages"],
        "repo_name": slug(topic) + "-radar",
        "readme_hook": f"Everything worth reading about {topic}, ranked and refreshed daily.",
        "profile_fit": {
            "score": 6,
            "reason": "Shows data plumbing + a shipped artifact, but it is a weaker story than a "
            "trend-timed tool. Use as a lead, not a finished spec.",
        },
        "risk": "Scraping rules and stale data if the source changes its API.",
        "sources": [{"title": top.title, "url": top.url}],
    }


def generate(
    profile_text: str,
    signals: list[Signal],
    avoid: list[str],
    provider: str | None = None,
) -> Idea:
    provider = (provider or os.environ.get("LLM_PROVIDER") or "auto").lower()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    prompt = build_prompt(profile_text, signals, avoid)

    order: list[str]
    if provider == "auto":
        order = [p for p in (("gemini", gemini_key), ("groq", groq_key)) if p[1]]
    else:
        order = [(provider, {"gemini": gemini_key, "groq": groq_key}.get(provider, ""))]

    for name, key in order:
        if not key:
            print(f"[llm] {name}: no key, skipping")
            continue
        try:
            fn = gemini if name == "gemini" else groq
            print(f"[llm] trying {name}")
            return from_json(fn(prompt, key), engine="llm")
        except Exception as exc:
            print(f"[llm] {name} failed: {exc}")

    print("[llm] falling back to deterministic template")
    return from_json(template(signals), engine="template")
