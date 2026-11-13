"""Entry point: collect -> score -> generate -> dedupe -> deliver -> log."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar import llm, notify, state
from radar import profile as profile_mod
from radar.idea import Idea, render_markdown, render_telegram
from radar.sources import Signal, collect, top_signals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="radar", description="One portfolio-grade idea per day.")
    p.add_argument("--dry-run", action="store_true", help="print instead of sending, write nothing")
    p.add_argument("--provider", choices=["auto", "gemini", "groq"], default=None)
    p.add_argument("--sources", default="", help="comma list, e.g. hn,github,arxiv")
    p.add_argument("--date", default="", help="override the date label (YYYY-MM-DD)")
    p.add_argument("--min-fit", type=int, default=6, help="regenerate if profile fit is below this")
    p.add_argument(
        "--if-missing",
        action="store_true",
        help="exit quietly if today already produced a real idea (for retry schedules)",
    )
    return p.parse_args(argv)


def pick_idea(
    profile_text: str,
    signals: list[Signal],
    records: list[dict],
    provider: str | None,
    min_fit: int,
) -> Idea:
    avoid = [r.get("title", "") for r in records if r.get("title")]
    idea = llm.generate(profile_text, signals, avoid, provider)

    for retry in range(2):
        dups = state.duplicates(idea.title, idea.one_liner, records)
        weak = idea.engine == "llm" and _fit(idea) < min_fit
        if not dups and not weak:
            return idea
        reason = f"duplicate of: {dups[0]}" if dups else "profile fit too low"
        print(f"[dedupe] retry {retry + 1} ({reason})")
        avoid.append(idea.title)
        signals = signals[: max(6, len(signals) // 2)]
        idea = llm.generate(profile_text, signals, avoid, provider)

    if state.duplicates(idea.title, idea.one_liner, records):
        print("[dedupe] final pick is still close to history; sending it anyway")
    return idea


def _fit(idea: Idea) -> int:
    try:
        return int(float(idea.profile_fit.get("score", 0)))
    except (TypeError, ValueError):
        return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if (stream.encoding or "").lower() not in ("utf-8", "utf8"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    day = args.date or state.today()
    only = [s.strip() for s in args.sources.split(",") if s.strip()] or None

    records = state.load()
    if args.if_missing and state.has_good_idea(records, day):
        print(f"[skip] {day} already has an LLM idea; nothing to do")
        return 0

    signals, errors = collect(only)
    if not signals:
        print(f"[fatal] no signals collected: {errors}")
        if not args.dry_run:
            notify.send(
                "<b>Idea Radar</b>\nNo signal today — every source failed. Check the Actions log."
            )
        return 1

    trimmed = top_signals(signals, 40)
    profile = profile_mod.fetch_profile()

    try:
        idea = pick_idea(
            profile_mod.summary(profile), trimmed, records, args.provider, args.min_fit
        )
    except Exception:
        traceback.print_exc()
        if not args.dry_run:
            notify.send("<b>Idea Radar</b>\nIdea generation failed today. Log: Actions run.")
        return 1

    message = render_telegram(idea, day, errors)
    payload = idea.as_dict()
    payload["date"] = day

    if args.dry_run:
        print("\n" + "=" * 72)
        print(message)
        print("=" * 72)
        print(f"[dry-run] engine={idea.engine} fit={_fit(idea)}/10 sources_ok={len(trimmed)}")
        return 0

    sent = notify.send(message)
    print(f"[notify] sent in {sent} message(s)")

    records = state.record(records, payload, day)
    state.save(records)
    path = state.append_markdown(payload, render_markdown(idea, day), day)
    print(f"[log] {path}")
    print(f"[done] {idea.title} (fit {_fit(idea)}/10, engine {idea.engine})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
