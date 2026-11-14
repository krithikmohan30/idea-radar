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
Return JSON only. No prose, no markdown fences. No emoji anywhere in the output."""

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


_MODEL_CACHE: dict[str, list[str]] = {}

# A single daily idea is not worth 14 minutes of hanging sockets. The model ladder
# provides the retries; each HTTP attempt just needs to be quick and bounded.
CALL_TIMEOUT = 40.0
ATTEMPTS_PER_MODEL = 2
DEADLINE_SECONDS = float(os.environ.get("LLM_DEADLINE_SECONDS", "240"))


def _post(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = CALL_TIMEOUT,
) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(ATTEMPTS_PER_MODEL):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json", **(headers or {})},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last = exc
        else:
            if resp.ok:
                return resp.json()
            last = LLMError(f"{resp.status_code}: {resp.text[:400]}")
            if resp.status_code not in {429, 500, 502, 503, 504}:
                raise last
        if attempt + 1 < ATTEMPTS_PER_MODEL:
            time.sleep(5)
    raise LLMError(str(last))


def _rank_model(name: str) -> tuple[int, int, float] | None:
    """Score a Gemini model name; higher is better. None = not usable for us.

    Family outranks version on purpose: pro models have no free tier, so a newer
    pro name would 429 in a free-tier key. Stable flash first, preview flash next,
    pro only as a last resort. Suffixed variants (-tts, -image, -audio) are skipped:
    they answer the same generateContent call but reject system instructions.
    """
    match = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-(flash-lite|flash|pro)(?:-(.+))?", name)
    if not match:
        return None
    suffix = match.group(3) or ""
    if suffix and not (suffix.isdigit() or suffix in {"preview", "exp", "latest"}):
        return None
    family = {"flash": 3, "pro": 2, "flash-lite": 1}[match.group(2)]
    stable = 0 if suffix in {"preview", "exp"} else 1
    return (family, stable, float(match.group(1)))


def resolve_models(api_key: str, refresh: bool = False) -> list[str]:
    """Ask the API which models this key may use, best first.

    Hard-coding a model name rots: Google retires names and returns a 404 telling
    you the replacement. Discovery means the bot keeps working without a code change.
    The list matters too: a free-tier model that is overloaded (503) can be skipped
    for the next one instead of losing the day.
    """
    if not refresh and _MODEL_CACHE.get(api_key[:8]):
        return list(_MODEL_CACHE[api_key[:8]])

    resp = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=1000",
        timeout=30,
    )
    resp.raise_for_status()
    names = [
        m["name"].split("/", 1)[-1]
        for m in resp.json().get("models", [])
        if "generateContent" in (m.get("supportedGenerationMethods") or [])
    ]
    ranked = sorted(((_rank_model(n), n) for n in names if _rank_model(n)), reverse=True)
    if not ranked:
        raise LLMError(f"no usable Gemini model in {names[:5]}")

    ordered = [name for _, name in ranked]
    _MODEL_CACHE[api_key[:8]] = ordered
    print(f"[llm] gemini candidates: {', '.join(ordered[:3])} (of {len(names)} available)")
    return ordered


def resolve_model(api_key: str, refresh: bool = False) -> str:
    return resolve_models(api_key, refresh)[0]


def _gemini_call(
    prompt: str,
    api_key: str,
    model: str,
    inline_system: bool = False,
    json_mode: bool = True,
    timeout: float = CALL_TIMEOUT,
) -> dict[str, Any]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    text = f"{SYSTEM}\n\n{prompt}" if inline_system else prompt
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 4096},
    }
    if not inline_system:
        body["systemInstruction"] = {"parts": [{"text": SYSTEM}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"

    data = _post(url, body, timeout=timeout)
    try:
        return load_json(data["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected Gemini payload: {json.dumps(data)[:300]}") from exc


def gemini(prompt: str, api_key: str, model: str | None = None) -> dict[str, Any]:
    explicit = model or os.environ.get("GEMINI_MODEL", "").strip()
    candidates = [explicit] if explicit else resolve_models(api_key)[:3]
    deadline = time.monotonic() + DEADLINE_SECONDS
    last: Exception | None = None

    for index, candidate in enumerate(candidates):
        remaining = deadline - time.monotonic()
        if remaining < 15:
            print("[llm] out of time budget, stopping the model ladder")
            break
        started = time.monotonic()
        try:
            return _gemini_model(prompt, api_key, candidate, deadline)
        except LLMError as exc:
            last = exc
            elapsed = time.monotonic() - started
            print(f"[llm] {candidate} failed after {elapsed:.0f}s: {str(exc)[:160]}")
            if index + 1 < len(candidates):
                print(f"[llm] trying {candidates[index + 1]}")
    raise LLMError(str(last))


def _gemini_model(prompt: str, api_key: str, model: str, deadline: float) -> dict[str, Any]:
    """One model, with the per-model fallbacks: system prompt, JSON mode, retirement."""
    inline_system = False
    json_mode = True

    for _ in range(3):
        remaining = deadline - time.monotonic()
        if remaining < 10:
            raise LLMError("time budget exhausted")
        timeout = max(10.0, min(CALL_TIMEOUT, remaining))
        try:
            return _gemini_call(prompt, api_key, model, inline_system, json_mode, timeout)
        except LLMError as exc:
            message = str(exc)
            if not inline_system and "Developer instruction" in message:
                print("[llm] model rejects systemInstruction; inlining the system prompt")
                inline_system = True
                continue
            if json_mode and "responseMimeType" in message:
                print("[llm] model rejects JSON mode; parsing free-form output")
                json_mode = False
                continue
            if "404" in message or "no longer available" in message:
                replacement = resolve_model(api_key, refresh=True)
                if replacement == model:
                    raise
                print(f"[llm] {model} retired, retrying with {replacement}")
                model = replacement
                continue
            raise

    raise LLMError(f"gemini gave up after retries against {model}")


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
