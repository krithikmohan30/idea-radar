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


def gemini_body(text: str = json.dumps(IDEA)) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_gemini_parses_response(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(200, gemini_body()))
    assert llm.gemini("prompt", "key")["title"] == "Spec Drift"


def test_gemini_retries_then_succeeds(monkeypatch):
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


def test_gemini_rejects_bad_key_without_retry(monkeypatch):
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        return FakeResp(400, {"error": "API key not valid"})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(llm.LLMError):
        llm.gemini("prompt", "bad")
    assert calls["n"] == 1


def test_groq_parses_response(monkeypatch):
    body = {"choices": [{"message": {"content": json.dumps(IDEA)}}]}
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(200, body))
    assert llm.groq("prompt", "key")["title"] == "Spec Drift"


def test_generate_falls_back_without_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    idea = llm.generate("profile", SIGNALS, [], None)
    assert idea.engine == "template"
    assert idea.repo_name


def test_generate_uses_gemini_when_key_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResp(200, gemini_body()))
    idea = llm.generate("profile", SIGNALS, ["old idea"], None)
    assert idea.engine == "llm"
    assert idea.title == "Spec Drift"


def test_generate_falls_back_when_provider_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
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
