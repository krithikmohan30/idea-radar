"""Tests that do not touch the network."""

from __future__ import annotations

from radar import notify, state
from radar.idea import from_json, load_json, render_markdown, render_telegram, slug
from radar.sources import Signal, top_signals

PAYLOAD = {
    "title": "Spec Drift: OpenAPI breaking-change detector",
    "one_liner": "Point it at two spec versions, get a severity-ranked list of breaking changes.",
    "why_now": "Three posts this week about API churn.",
    "build_plan": {"Day 1": "parse + diff", "Day 2": "CLI + CI action"},
    "stack": ["Python", "Click", "pytest"],
    "repo_name": "spec-drift",
    "readme_hook": "Catch breaking API changes before your users do.",
    "profile_fit": {"score": 9, "reason": "No dev-tooling repo yet."},
    "risk": "Spec edge cases",
    "sources": [{"title": "HN thread", "url": "https://example.com/a"}],
}


def test_slug():
    assert slug("Hello, World! -- 2026") == "hello-world-2026"
    assert slug("???") == "idea"


def test_from_json_fills_defaults():
    idea = from_json(
        {
            "title": "Spec Drift",
            "one_liner": "O",
            "why_now": "W",
            "build_plan": {"Day 1": "x"},
            "stack": "Python",
        }
    )
    assert idea.repo_name == "spec-drift"
    assert idea.stack == ["Python"]


def test_from_json_requires_build_plan():
    try:
        from_json({"title": "T", "one_liner": "O", "why_now": "W", "build_plan": []})
    except ValueError as exc:
        assert "build_plan" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_from_json_rejects_missing_fields():
    try:
        from_json({"title": "T"})
    except ValueError as exc:
        assert "one_liner" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_plan_from_list():
    idea = from_json(
        {
            "title": "T",
            "one_liner": "O",
            "why_now": "W",
            "build_plan": ["a", "b"],
            "stack": "Python",
        }
    )
    assert list(idea.build_plan) == ["Day 1", "Day 2"]


def test_render_telegram_escapes_html():
    idea = from_json({**PAYLOAD, "title": "A <script> B"})
    out = render_telegram(idea, "2026-11-13")
    assert "&lt;script&gt;" in out
    assert "<script>" not in out
    assert "9/10" in out
    assert "https://example.com/a" in out


def test_render_telegram_flags_fallback():
    idea = from_json(PAYLOAD)
    idea.engine = "template"
    assert "Fallback engine" in render_telegram(idea, "2026-11-13")
    idea.engine = "llm"
    assert "Fallback engine" not in render_telegram(idea, "2026-11-13")


def test_render_markdown():
    md = render_markdown(from_json(PAYLOAD), "2026-11-13")
    assert md.startswith("## 2026-11-13 — Spec Drift")
    assert "spec-drift" in md


def test_load_json_handles_fences_and_prose():
    assert load_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert load_json('Sure! {"a": 2} hope that helps') == {"a": 2}


def test_chunk_respects_limit():
    text = "\n".join(f"line {i} " + "x" * 80 for i in range(200))
    chunks = notify.chunk(text, limit=1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_similar_detects_near_duplicates():
    assert state.similar(
        "OpenAPI breaking change detector", "OpenAPI breaking changes detector tool"
    )
    assert not state.similar("OpenAPI diff tool", "A pomodoro timer for plants")


def test_duplicates_window():
    records = [{"title": "Telegram budget bot", "one_liner": "Track spends via chat"}]
    assert state.duplicates("Telegram budget bot", "Track spends via chat", records)
    assert not state.duplicates("Spec drift detector", "API diff", records)


def test_top_signals_dedupes_and_ranks():
    signals = [
        Signal("hn", "Same title", "u1", score=10),
        Signal("hn", "Same title", "u2", score=99),
        Signal("hn", "Other", "u3", score=50, comments=5),
    ]
    out = top_signals(signals, 10)
    assert len(out) == 2
    assert out[0].url == "u2"


def test_top_signals_interleaves_sources():
    signals = [Signal("hn", f"hn {i}", f"h{i}", score=1000 - i) for i in range(5)]
    signals += [Signal("arxiv", f"paper {i}", f"p{i}", score=1) for i in range(5)]
    out = top_signals(signals, limit=4)
    assert {s.source for s in out} == {"hn", "arxiv"}


def test_append_markdown_creates_month_file(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "LOG_DIR", tmp_path / "ideas")
    md = render_markdown(from_json(PAYLOAD), "2026-11-13")
    path = state.append_markdown(PAYLOAD, md, "2026-11-13")
    assert path.name == "2026-11.md"
    text = path.read_text(encoding="utf-8")
    assert "Idea log — 2026-11" in text
    assert "Spec Drift" in text


def test_append_markdown_replaces_same_day(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "LOG_DIR", tmp_path / "ideas")
    first = render_markdown(from_json({**PAYLOAD, "title": "Old Idea"}), "2026-11-13")
    second = render_markdown(from_json(PAYLOAD), "2026-11-13")
    path = state.append_markdown(PAYLOAD, first, "2026-11-13")
    state.append_markdown(PAYLOAD, second, "2026-11-13")
    text = path.read_text(encoding="utf-8")
    assert text.count("## 2026-11-13") == 1
    assert "Old Idea" not in text
    assert "Spec Drift" in text


def test_append_markdown_keeps_other_days(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "LOG_DIR", tmp_path / "ideas")
    path = state.append_markdown(
        PAYLOAD, render_markdown(from_json(PAYLOAD), "2026-11-12"), "2026-11-12"
    )
    state.append_markdown(PAYLOAD, render_markdown(from_json(PAYLOAD), "2026-11-13"), "2026-11-13")
    text = path.read_text(encoding="utf-8")
    assert text.count("## 2026-11-") == 2
    assert text.index("2026-11-12") < text.index("2026-11-13")


def test_record_replaces_same_day():
    records = [{"date": "2026-11-13", "title": "old"}]
    records = state.record(records, PAYLOAD, "2026-11-13")
    assert len(records) == 1
    assert records[0]["title"] == PAYLOAD["title"]


def test_has_good_idea_only_counts_llm_entries():
    template = [{"date": "2026-11-13", "engine": "template"}]
    real = [{"date": "2026-11-13", "engine": "llm"}]
    assert not state.has_good_idea(template, "2026-11-13")
    assert state.has_good_idea(real, "2026-11-13")
    assert not state.has_good_idea(real, "2026-11-14")
