"""Provider wiring, with HTTP mocked so no key or quota is needed."""

from __future__ import annotations

import json

import pytest
import requests

from radar import llm
from radar.sources import Signal

IDEA = {
    "title": "Spec Drift",
    "one_liner": "Diff two OpenAPI specs by severity.",
    "why_now": "API churn is loud this week.",
    "build_plan": {"Day 1": "parse", "Day 2": "ship"},
    "stack": ["Python"],
    "repo_name": "spec-drift",
    "profile_fit": {"score": 9, "reason": "fills a gap"},
    "sources": [{"title": "HN", "url": "https://example.com"}],
}

SIGNALS = [
    Signal(
        "hn",
        "OpenAPI breaking changes are everywhere",
        "https://example.com",
        score=200,
        comments=40,
    )
]


class FakeResp:
    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload
        self.ok = 200 <= status < 300
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(self.text, response=None)


def gemini_body(text: str = json.dumps(IDEA)) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_gemini_parses_response(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(200, gemini_body()))
    assert llm.gemini("prompt", "key")["title"] == "Spec Drift"


def test_gemini_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            return FakeResp(429, {"error": "rate limited"})
        return FakeResp(200, gemini_body())

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    assert llm.gemini("prompt", "key")["repo_name"] == "spec-drift"
    assert calls["n"] == 2


def test_resolve_model_prefers_newest_stable_flash(monkeypatch):
    llm._MODEL_CACHE.clear()
    models = {
        "models": [
            {"name": "models/gemini-1.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-5.0-pro", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
            {
                "name": "models/gemini-4.0-flash-preview",
                "supportedGenerationMethods": ["generateContent"],
            },
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
        ]
    }
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: FakeResp(200, models))
    assert llm.resolve_model("abc12345") == "gemini-3.8-flash"


def test_resolve_model_skips_tts_and_image_variants(monkeypatch):
    llm._MODEL_CACHE.clear()
    models = {
        "models": [
            {
                "name": "models/gemini-3.8-flash-tts",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-3.8-flash-image",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-3.5-flash",
                "supportedGenerationMethods": ["generateContent"],
            },
        ]
    }
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: FakeResp(200, models))
    assert llm.resolve_model("zzz99999") == "gemini-3.5-flash"


def test_gemini_inlines_system_prompt_when_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    seen = []

    def fake_post(url, **kwargs):
        body = kwargs["json"]
        seen.append("systemInstruction" in body)
        if "systemInstruction" in body:
            return FakeResp(
                400, {"error": {"message": "Developer instruction is not enabled for this model"}}
            )
        return FakeResp(200, gemini_body())

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.gemini("prompt", "key")["title"] == "Spec Drift"
    assert seen == [True, False]


def test_gemini_moves_to_next_model_when_overloaded(monkeypatch):
    llm._MODEL_CACHE.clear()
    models = {
        "models": [
            {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
        ]
    }
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: FakeResp(200, models))
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    def fake_post(url, **kwargs):
        if "gemini-3.8-flash" in url:
            return FakeResp(503, {"error": {"message": "high demand"}})
        return FakeResp(200, gemini_body())

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.gemini("prompt", "qqq11111")["title"] == "Spec Drift"


def test_gemini_gives_up_when_out_of_time(monkeypatch):
    llm._MODEL_CACHE.clear()
    models = {
        "models": [
            {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]}
        ]
    }
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: FakeResp(200, models))
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(llm, "DEADLINE_SECONDS", 0.0)
    with pytest.raises(llm.LLMError):
        llm.gemini("prompt", "ddd44444")


def test_post_stops_after_two_attempts(monkeypatch):
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        return FakeResp(503, {"error": "high demand"})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    with pytest.raises(llm.LLMError):
        llm._post("https://example.com", {"a": 1})
    assert calls["n"] == 2


def test_gemini_recovers_from_retired_model(monkeypatch):
    llm._MODEL_CACHE.clear()
    models = {
        "models": [
            {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]}
        ]
    }
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: FakeResp(200, models))
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)

    def fake_post(url, *args, **kwargs):
        if "gemini-2.5-flash" in url:
            return FakeResp(404, {"error": {"message": "no longer available to new users"}})
        return FakeResp(200, gemini_body())

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    assert llm.gemini("prompt", "abc12345")["title"] == "Spec Drift"


def test_gemini_rejects_bad_key_without_retry(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        return FakeResp(400, {"error": "API key not valid"})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(llm.LLMError):
        llm.gemini("prompt", "bad")
    assert calls["n"] == 1


def test_groq_parses_response(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    body = {"choices": [{"message": {"content": json.dumps(IDEA)}}]}
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(200, body))
    assert llm.groq("prompt", "key")["title"] == "Spec Drift"


def test_template_prefers_buildable_sources_and_cleans_titles():
    signals = [
        Signal("hn", "Company raises $50M", "https://example.com/news", score=900),
        Signal(
            "github",
            "acme/gestalt: MobileGestalt editor for iOS 27.x, featuring presets",
            "https://github.com/acme/gestalt",
            score=7430,
        ),
    ]
    payload = llm.template(signals)
    assert "gestalt" in payload["repo_name"]
    assert "7,430 stars" in payload["why_now"]
    assert "acme/" not in payload["title"]
    assert len(payload["sources"]) == 2


def test_clean_topic_strips_prefixes_and_clauses():
    assert (
        llm._clean_topic("Show HN: Desk Widgets \u2013 drag and drop")
        == "Desk Widgets \u2013 drag and drop"
    )
    assert llm._clean_topic("acme/repo: A tool, featuring things") == "A tool"
    assert llm._clean_topic("") == "today's top signal"


def test_template_rotates_framings():
    a = llm.template([Signal("github", "alpha beta gamma", "u1", score=10)])
    b = llm.template([Signal("github", "zeta omega delta", "u2", score=10)])
    assert a["title"] != b["title"]


def test_resolve_groq_model_prefers_known_family(monkeypatch):
    llm._MODEL_CACHE.clear()
    models = {
        "data": [
            {"id": "whisper-large-v3"},
            {"id": "llama-3.3-70b-versatile"},
            {"id": "llama-3.1-8b-instant"},
        ]
    }
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: FakeResp(200, models))
    assert llm.resolve_groq_model("ggg11111") == "llama-3.3-70b-versatile"


def test_groq_uses_discovered_model(monkeypatch):
    llm._MODEL_CACHE.clear()
    models = {"data": [{"id": "openai/gpt-oss-120b"}]}
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: FakeResp(200, models))
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    sent = {}

    def fake_post(url, **kwargs):
        sent["model"] = kwargs["json"]["model"]
        return FakeResp(200, {"choices": [{"message": {"content": json.dumps(IDEA)}}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.groq("prompt", "hhh22222")["title"] == "Spec Drift"
    assert sent["model"] == "openai/gpt-oss-120b"


def test_generate_falls_back_without_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    idea = llm.generate("profile", SIGNALS, [], None)
    assert idea.engine == "template"
    assert idea.repo_name


def test_generate_uses_gemini_when_key_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(200, gemini_body()))
    idea = llm.generate("profile", SIGNALS, ["old idea"], None)
    assert idea.engine == "llm"
    assert idea.title == "Spec Drift"


def test_generate_falls_back_when_provider_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)

    def boom(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(llm.requests, "post", boom)
    idea = llm.generate("profile", SIGNALS, [], "gemini")
    assert idea.engine == "template"


def test_prompt_includes_candidates_and_avoid_list():
    prompt = llm.build_prompt("PROFILE-MARK", SIGNALS, ["Do not repeat me"])
    assert "PROFILE-MARK" in prompt
    assert "https://example.com" in prompt
    assert "Do not repeat me" in prompt


def test_template_produces_a_four_entry_shortlist():
    signals = [
        Signal("github", "alpha one", "https://example.com/a", score=10),
        Signal("hn", "beta two", "https://example.com/b", score=9),
        Signal("lobsters", "gamma three", "https://example.com/c", score=8),
        Signal("devto", "delta four", "https://example.com/d", score=7),
    ]
    payload = llm.template(signals)
    assert len(payload["shortlist"]) == 4
    assert payload["shortlist"][0]["rank"] == 1
    assert payload["shortlist"][0]["title"] == "alpha one"


def test_prompt_asks_for_a_cross_source_shortlist():
    prompt = llm.build_prompt("profile", [Signal("hn", "x", "https://e.com/x")], [])
    assert '"shortlist"' in prompt
    assert "different source" in prompt
    assert "A 10 is not allowed" in llm.SYSTEM
