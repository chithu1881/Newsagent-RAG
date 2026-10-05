"""
Smoke test for the RAG engine - quick pass/fail checks you can run after every change.

    venv\\Scripts\\python -m eval.smoke_test

Checks:
  - query understanding: time ranges and categories are inferred correctly (no models needed)
  - grounding: off-topic questions get "I don't have news on that." with no sources
  - answering: on-topic questions return sources, inside the asked time range, and cite them
Runs on the frozen 5 Oct 2026 knowledge base (eval/kb_snapshot) with a frozen clock, so results don't drift.
The full 25-question evaluation is eval/run_eval.py.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ["KB_SOURCE"] = "local"
os.environ["CHROMA_DIR"] = str(Path(__file__).resolve().parent / "kb_snapshot")

from rag.engine import NO_NEWS, answer, understand  # noqa: E402

NOW = datetime(2026, 10, 5, 15, 0)      # for the pure understanding checks
EVAL_NOW = datetime(2026, 10, 5, 23, 0)  # the snapshot's "now" for the answering checks
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail and not ok else ""))


# ---- 1. understanding (pure rules) ------------------------------------------
u = understand("What did the RBI announce this week?", NOW)
check("'this week' -> last 7 days", u["date_from"] == NOW - timedelta(days=7), u)
check("RBI -> finance", u["categories"] == ["finance"], u["categories"])
u = understand("What happened in Parliament yesterday?", NOW)
check("'yesterday' -> yesterday only", u["date_from"] == datetime(2026, 10, 4) and u["date_to"] == datetime(2026, 10, 5), u)
check("Parliament -> politics", u["categories"] == ["politics"], u["categories"])
u = understand("AI chip news in the last 3 days", NOW)
check("'last 3 days' parsed", u["date_from"] == NOW - timedelta(days=3), u)
check("topic strips time words", u["topic"] == "AI chip", u["topic"])
u = understand("Tell me something interesting", NOW)
check("no time words -> no date filter", u["date_from"] is None and u["date_to"] is None, u)

# ---- 2. grounding: must refuse ------------------------------------------------
for q in ["Who won the IPL final?", "What is a good recipe for paneer butter masala?",
          "What did NASA announce about Mars in 1997?"]:
    r = answer(q, now=EVAL_NOW)
    check(f"refuses: {q}", r["answer"].startswith(NO_NEWS[:-1]) and not r["sources"], r["answer"][:80])

# ---- 3. answering: must find sources in range ------------------------------
for q, cat in [("What is happening with bond yields and the RBI?", "finance"),
               ("Any news about AI companies this week?", "tech"),
               ("What are the main political stories this week?", "politics")]:
    r = answer(q, now=EVAL_NOW)
    srcs = r["sources"]
    check(f"answers: {q}", bool(srcs), r["answer"][:80])
    if srcs:
        week_ago = (EVAL_NOW - timedelta(days=7)).timestamp()
        check("   sources are within the asked week", all(s["published_ts"] >= week_ago for s in srcs if "week" in q))
        check(f"   top source is {cat}", srcs[0]["category"] == cat, srcs[0]["category"])
        check("   answer cites [n]", "[" in r["answer"], r["answer"][:80])

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
