# Idea Radar

One buildable project idea in my Telegram every day, mined from that day's
actual tech signal and scored against my own GitHub profile so it never suggests things
I have already built.

This repo is the bot. It is also the point: instead of a static "ideas.md", the commit
history is a dated log of what I noticed and what I decided to build.

## The loop

| Stage | What happens | File |
| --- | --- | --- |
| Collect | Six public sources, no scraping, each isolated so one outage does not matter | `src/radar/sources.py` |
| Reflect | Reads my public repos + profile README, so duplicate suggestions are blocked | `src/radar/profile.py` |
| Score | Ranks four candidates from four different feeds, specs the winner, grades fit against a rubric | `src/radar/llm.py` |
| Dedupe | Token overlap against the last 60 ideas, plus every link cited in the last 21 days | `src/radar/state.py` |
| Deliver | Telegram message, HTML, chunked safely | `src/radar/notify.py` |
| Log | Appends to `ideas/YYYY-MM.md` and commits | `src/radar/main.py` |

Sources: Hacker News front page, Show HN, newly trending GitHub repos, arXiv cs.AI/cs.SE,
Lobsters, dev.to.

Every run produces exactly one idea, and shows its work: the four candidates it ranked
before choosing, and the score it gave each. Links already used in the last three weeks are
removed before the model sees the pool, so a write-up that stays on a front page for a week
cannot keep supplying new proposals. Scores are calibrated in code, not trusted: a 10 becomes
a 9, and a shortlist where every candidate scored 8+ caps the winner at 8.

## What a delivery looks like

```
Idea Radar · 2026-11-14

Spec Drift: catch breaking API changes before your users do
Point it at two OpenAPI specs, get a severity-ranked list of what breaks and who is affected.

Why now
Three separate threads this week on API churn after a major framework release...

2-day build
  Day 1  Parse + normalise both specs, diff endpoints, rank by severity
  Day 2  CLI output, GitHub Action wrapper, README with a real before/after

Stack      Python · Click · pytest
Repo name  spec-drift

Profile fit 9/10 — No developer-tooling repo yet; proves library design, not just app wiring.
```

## Why it does not just say "build an AI app"

- Hard filters in the system prompt: two evenings max, visible demo in 30 seconds,
  no clones, no todo apps, no thin model wrappers.
- The developer's real repo list is in the prompt as a negative constraint.
- `profile_fit` below 6 triggers a regenerate with the rejected title added to the avoid list.
- Everything is logged, so repeated suggestions are visible in the commit history.

## Free by construction

- **Schedule:** GitHub Actions cron, so it runs whether or not my laptop is awake.
- **Model:** Gemini free tier (Groq as a second option). One call per day.
- **Fallback:** if every key fails, a deterministic template still produces a lead from the
  top signal, labelled as a fallback. No silent empty days.
- **Cost:** zero. No database, no hosting, no queue. State is a JSON file in the repo.

## Setup

### 1. Telegram

1. Message `@BotFather` → `/newbot` → copy the token.
2. Send your new bot any message, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `message.chat.id`.

### 2. Free model keys

- Gemini: <https://aistudio.google.com/apikey> → create key.
- Groq (recommended backup): <https://console.groq.com/keys>.

Two providers matter because free tiers run out. `LLM_PROVIDER=auto` tries Gemini,
then Groq, then the deterministic fallback.

### 3. Repo secrets

```powershell
gh secret set TELEGRAM_BOT_TOKEN --repo <you>/idea-radar
gh secret set TELEGRAM_CHAT_ID    --repo <you>/idea-radar
gh secret set GEMINI_API_KEY      --repo <you>/idea-radar
gh secret set GROQ_API_KEY        --repo <you>/idea-radar   # optional backup
```

Verify the wiring before trusting the schedule:

```powershell
gh workflow run check.yml --repo <you>/idea-radar
gh run watch --repo <you>/idea-radar
```

### 4. Test before you trust it

```powershell
gh workflow run daily.yml --repo <you>/idea-radar -f dry_run=true
gh run watch --repo <you>/idea-radar
```

Then a real run: `gh workflow run daily.yml`. After that the cron fires daily at 01:30 UTC
with a 04:30 UTC retry. Change the cron in `.github/workflows/daily.yml` for another timezone.

**On timing, honestly:** GitHub queues `schedule` runs behind interactive traffic and delays
them on low-activity repos. In practice these have landed 6-11 hours after the cron time, so
treat the schedule as "once a day", not "at 07:00". A manual `gh workflow run` starts within
seconds. If punctual delivery ever matters, an external pinger calling the `workflow_dispatch`
API is the reliable fix; the cron alone will not do it.

## Local

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m radar.main --dry-run
```

`--dry-run` prints the message and writes nothing. Flags: `--provider gemini|groq|auto`,
`--sources hn,github,arxiv`, `--date 2026-11-14`, `--min-fit 7`.

Tests and lint:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest ruff
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

## Layout

```
src/radar/
  sources.py   six fetchers, each failure-isolated
  profile.py   my repos + README, cached to data/profile.json
  llm.py       Gemini, Groq, deterministic fallback
  idea.py      schema normalisation + Telegram/Markdown rendering
  state.py     history, similarity, monthly log
  notify.py    Telegram delivery
  main.py      orchestrator
ideas/         the dated log this repo exists to produce
data/          history + profile cache
tools/         wiring diagnostics (telegram_check.py, provider_check.py)
```

## Ideas accepted so far

See [`ideas/`](ideas/) — one entry per day, newest last.
