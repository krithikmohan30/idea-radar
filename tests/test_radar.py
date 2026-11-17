"""Tests that do not touch the network."""

from __future__ import annotations

from radar import notify, state
from radar.idea import (
    calibrate,
    fit_score,
    from_json,
    load_json,
    render_markdown,
    render_telegram,
    slug,
)
from radar.sources import (
    Signal,
    collapse_duplicates,
    drop_seen,
    normalize_url,
    top_signals,
)

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


# --- source diversity -------------------------------------------------------


def _sig(source: str, title: str, url: str, score: int = 0, comments: int = 0) -> Signal:
    return Signal(source, title, url, score=score, comments=comments)


def test_normalize_url_ignores_scheme_www_and_tracking():
    assert normalize_url("https://www.Example.com/a/?utm_source=hn#top") == "example.com/a"
    assert normalize_url("http://example.com/a") == "example.com/a"
    assert normalize_url("") == ""


def test_collapse_duplicates_keeps_the_loudest_copy():
    pool = [
        _sig("hn", "MCP is dead simple", "https://example.com/mcp?ref=hn", score=10),
        _sig("lobsters", "MCP is dead simple", "https://example.com/mcp", score=90, comments=4),
        _sig("github", "Something else", "https://github.com/x/y", score=5),
    ]
    out = collapse_duplicates(pool)
    assert len(out) == 2
    kept = next(s for s in out if "example.com/mcp" in s.url)
    assert kept.source == "lobsters"


def test_drop_seen_removes_previously_cited_links():
    pool = [
        _sig("hn", "Old story", "https://example.com/old"),
        _sig("hn", "New story", "https://example.com/new"),
    ]
    kept, dropped = drop_seen(pool, {"example.com/old"})
    assert dropped == 1
    assert [s.title for s in kept] == ["New story"]


def test_drop_seen_is_a_noop_without_history():
    pool = [_sig("hn", "A", "https://example.com/a")]
    kept, dropped = drop_seen(pool, set())
    assert dropped == 0
    assert len(kept) == 1


def test_record_keeps_source_urls():
    records = state.record([], {**PAYLOAD, "engine": "llm"}, "2026-11-14")
    assert records[0]["sources"] == ["https://example.com/a"]


def test_recent_urls_respects_the_window_and_normalises():
    records = [
        {"date": "2026-11-14", "sources": ["https://www.Example.com/x?utm=1"]},
        {"date": "2026-09-01", "sources": ["https://example.com/old"]},
    ]
    used = state.recent_urls(records, days=21, today_iso="2026-11-17")
    assert "example.com/x" in used
    assert "example.com/old" not in used


def test_recent_urls_is_empty_without_history():
    assert state.recent_urls([]) == set()


# --- score calibration ------------------------------------------------------


def test_calibrate_caps_a_perfect_score():
    idea = from_json({**PAYLOAD, "profile_fit": {"score": 10, "reason": "flawless"}})
    assert fit_score(calibrate(idea)) == 9


def test_calibrate_caps_when_the_shortlist_has_no_spread():
    payload = {
        **PAYLOAD,
        "profile_fit": {"score": 9, "reason": "extends reaper"},
        "shortlist": [
            {"rank": i, "title": f"candidate {i}", "source": "hn", "score": 9} for i in range(1, 5)
        ],
    }
    assert fit_score(calibrate(from_json(payload))) == 8


def test_calibrate_leaves_a_genuinely_ranked_shortlist_alone():
    payload = {
        **PAYLOAD,
        "profile_fit": {"score": 9, "reason": "extends reaper"},
        "shortlist": [
            {"rank": 1, "title": "a", "source": "hn", "score": 9},
            {"rank": 2, "title": "b", "source": "github", "score": 6},
            {"rank": 3, "title": "c", "source": "devto", "score": 5},
        ],
    }
    assert fit_score(calibrate(from_json(payload))) == 9


def test_from_json_unwraps_pick_and_keeps_the_shortlist():
    idea = from_json(
        {
            "shortlist": [{"rank": 1, "title": "Alpha", "source": "hn", "score": 7}],
            "pick": PAYLOAD,
        }
    )
    assert idea.title == PAYLOAD["title"]
    assert [c["title"] for c in idea.shortlist] == ["Alpha"]


def test_markdown_shows_the_ranking():
    idea = from_json(
        {
            **PAYLOAD,
            "shortlist": [{"rank": 1, "title": "Alpha", "source": "hn", "score": 7}],
        }
    )
    md = render_markdown(idea, "2026-11-14")
    assert "**Ranked:**" in md
    assert "1. **Alpha**" in md


def test_from_json_finds_the_idea_under_any_wrapper_key():
    wrapped = {"shortlist": [{"title": "A", "score": 7}], "idea": PAYLOAD}
    assert from_json(wrapped).title == PAYLOAD["title"]

    odd = {"shortlist": [], "recommended_project": PAYLOAD}
    assert from_json(odd).title == PAYLOAD["title"]
