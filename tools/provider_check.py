"""Check which model provider actually answers right now.

Never prints keys. Exits non-zero only if no provider is usable, so it is safe to
run before trusting a scheduled job.
"""

from __future__ import annotations

import os
import sys

from radar import llm

PROMPT = 'Return JSON: {"status": "ok", "note": "one short sentence"}'


def check_gemini() -> bool:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("gemini: no key")
        return False
    try:
        models = llm.resolve_models(key)
    except Exception as exc:
        print(f"gemini: cannot list models ({type(exc).__name__}: {str(exc)[:120]})")
        return False
    print(f"gemini models visible: {len(models)} (top: {', '.join(models[:3])})")
    for name in models[:5]:
        try:
            llm._gemini_call(PROMPT, key, name, timeout=25)
        except llm.LLMError as exc:
            print(f"  {name}: {str(exc)[:90]}")
            continue
        print(f"  {name}: OK")
        return True
    return False


def check_groq() -> bool:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("groq: no key")
        return False
    try:
        model = llm.resolve_groq_model(key)
        llm.groq(PROMPT, key, model=model)
    except Exception as exc:
        print(f"groq: unusable ({type(exc).__name__}: {str(exc)[:120]})")
        return False
    print(f"groq: OK ({model})")
    return True


def main() -> int:
    ok = check_gemini()
    ok = check_groq() or ok
    print("verdict:", "at least one provider works" if ok else "no provider works")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
