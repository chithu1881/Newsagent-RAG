"""
Phase 5 - Evaluation of the RAG engine on eval/questions.csv (25 questions).

    venv\\Scripts\\python -m eval.run_eval                 # all questions
    venv\\Scripts\\python -m eval.run_eval --ids Q01 Q21   # just some

Every question is answered with a FROZEN clock (EVAL_NOW), so "today" / "this week" always mean the same
dates and articles collected later are ignored - re-runs are comparable while the news keeps flowing in.

Automatic checks (no LLM needed)
  refusal_ok      unanswerable -> must say "I don't have news on that."; others must not
  source_hit      an expected article is among the cited sources        (citation correctness)
  source_rank     its position (1 = top)
  fact_recall     share of expected key facts that appear in the answer (accuracy proxy)
  citations_ok    every [n] points to a real source and the answer cites at least one
  recency_ok      time-sensitive questions: every source is inside the asked time window

LLM-judged metrics (only when an LLM key is set) - same method as RAGAS, implemented here because the
ragas package does not install on Python 3.14:
  faithfulness      answer split into claims; share of claims supported by the retrieved text
  answer_relevancy  LLM writes 3 questions the answer would answer; mean cosine similarity to the asked one
  judge_correctness 1-5 grade of the answer against the reference answer

Output in eval/results/: run_<time>.csv (one row per question, with empty manual-score columns for you),
run_<time>.json (everything incl. retrieved text) and summary_<time>.md.
"""

import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
# The questions were written for the 5 Oct 2026 knowledge base, frozen in eval/kb_snapshot - never the live
# (cloud) store, whose contents change every 3 hours and lose old news after RETENTION_DAYS.
os.environ["CHROMA_MODE"] = "local"
os.environ["CHROMA_DIR"] = str(EVAL_DIR / "kb_snapshot")

from rag import engine, llm  # noqa: E402
from rag.engine import NO_NEWS, answer, understand  # noqa: E402

RESULTS_DIR = EVAL_DIR / "results"
EVAL_NOW = datetime(2026, 10, 5, 23, 0)     # the knowledge-base snapshot the questions were written for


# ---- automatic checks ------------------------------------------------------------

def norm(text):
    return re.sub(r"(?<=\d),(?=\d)", "", text.lower())     # "17,500" -> "17500"


def split_alts(cell):
    return [x.strip() for x in (cell or "").split("|") if x.strip()]


def auto_checks(q, result):
    text, sources = result["answer"], result["sources"]
    refused = text.strip().startswith(NO_NEWS[:-1])
    should_refuse = q["type"] == "unanswerable"
    row = {"refused": refused, "refusal_ok": refused == should_refuse}
    if should_refuse:
        return row

    expected = [e.lower() for e in split_alts(q["expected_source"])]
    rank = next((i + 1 for i, s in enumerate(sources) if any(e in s["title"].lower() for e in expected)), None)
    facts = split_alts(q["expected_facts"])
    found = [f for f in facts if norm(f) in norm(text)]
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    row.update(
        source_hit=rank is not None, source_rank=rank,
        fact_recall=round(len(found) / len(facts), 2) if facts else None,
        facts_missing="; ".join(f for f in facts if f not in found),
        citations_ok=bool(cited) and cited <= {s["n"] for s in sources},
    )
    if q["type"] == "time_sensitive":
        window = understand(q["question"], EVAL_NOW)
        lo = window["date_from"].timestamp() if window["date_from"] else 0
        hi = (window["date_to"] or EVAL_NOW).timestamp()
        row["recency_ok"] = bool(sources) and all(lo <= s["published_ts"] <= hi for s in sources)
    return row


# ---- LLM-judged metrics (RAGAS method) ----------------------------------------------

def parse_json(text):
    m = re.search(r"(\[.*\]|\{.*\})", text or "", re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return None


FAITH_SYSTEM = """You check whether an answer is supported by its context. Steps:
1. Break the ANSWER into short, self-contained factual claims (ignore citation markers like [1]).
2. For each claim decide if the CONTEXT directly supports it (true) or not (false). Use only the context.
Reply with JSON only: [{"claim": "...", "supported": true}, ...]"""

RELEVANCY_SYSTEM = """Write 3 different questions that the given answer would be a direct answer to.
Reply with JSON only: ["question 1", "question 2", "question 3"]"""

JUDGE_SYSTEM = """You grade a news assistant's answer against a reference answer written by a human.
Score 1-5: 5 = all key facts of the reference, nothing wrong; 4 = minor omission; 3 = partly right or
important omission; 2 = mostly wrong or missing; 1 = wrong or invented facts. Extra correct detail is fine.
If the reference is "I don't have news on that." then 5 = the answer declines, 1 = it answers anyway.
Reply with JSON only: {"score": n, "reason": "one sentence"}"""


def context_text(result):
    return "\n\n".join(f"[{s['n']}] {s['title']} ({s['date']})\n" + "\n".join(s["chunks"]) for s in result["sources"])


def faithfulness(question, result):
    if not result["sources"]:
        return None, []
    reply = llm.chat(FAITH_SYSTEM, f"QUESTION: {question}\n\nCONTEXT:\n{context_text(result)}\n\nANSWER:\n{result['answer']}",
                     max_tokens=2000)
    claims = parse_json(reply)
    if not isinstance(claims, list) or not claims:
        return None, []
    supported = sum(bool(c.get("supported")) for c in claims if isinstance(c, dict))
    return round(supported / len(claims), 2), [c.get("claim", "") for c in claims if isinstance(c, dict) and not c.get("supported")]


def answer_relevancy(question, result):
    if result["answer"].strip().startswith(NO_NEWS[:-1]):
        return 0.0
    generated = parse_json(llm.chat(RELEVANCY_SYSTEM, result["answer"], max_tokens=300))
    if not isinstance(generated, list) or not generated:
        return None
    vectors = engine._embedder().encode([question] + [str(g) for g in generated], normalize_embeddings=True)
    return round(float(statistics.mean(float(vectors[0] @ v) for v in vectors[1:])), 2)


def judge(q, result):
    reply = parse_json(llm.chat(JUDGE_SYSTEM, f"QUESTION: {q['question']}\n\nREFERENCE ANSWER: {q['reference_answer']}"
                                               f"\n\nASSISTANT ANSWER:\n{result['answer']}", max_tokens=300))
    if isinstance(reply, dict) and "score" in reply:
        return int(reply["score"]), reply.get("reason", "")
    return None, ""


# ---- run --------------------------------------------------------------------------------

def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def pct(values):
    values = [v for v in values if v is not None]
    return f"{sum(values)}/{len(values)} ({100 * sum(values) / len(values):.0f}%)" if values else "n/a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="only these question ids")
    args = parser.parse_args()

    with (EVAL_DIR / "questions.csv").open(encoding="utf-8") as f:
        questions = [q for q in csv.DictReader(f) if not args.ids or q["id"] in args.ids]
    use_llm = llm.available()
    mode = llm.describe()
    print(f"Evaluating {len(questions)} questions | clock frozen at {EVAL_NOW:%d %b %Y %H:%M} | {mode}")
    if not use_llm:
        print("No LLM key: answers are extractive and faithfulness / relevancy / judge scores are skipped.\n")

    rows, full = [], []
    for q in questions:
        result = answer(q["question"], now=EVAL_NOW)
        row = {"id": q["id"], "type": q["type"], "domain": q["domain"], "question": q["question"],
               **auto_checks(q, result),
               "best_relevance": result["info"]["best_relevance"], "seconds": result["info"]["seconds"]}
        if use_llm:
            row["faithfulness"], unsupported = faithfulness(q["question"], result)
            row["unsupported_claims"] = " | ".join(unsupported)
            row["answer_relevancy"] = answer_relevancy(q["question"], result)
            row["judge_correctness"], row["judge_reason"] = judge(q, result)
        row.update(answer=result["answer"], cited_sources=" || ".join(f"[{s['n']}] {s['title']} ({s['date']})"
                                                                         for s in result["sources"]),
                   reference_answer=q["reference_answer"], manual_accuracy_1to5="", manual_notes="")
        rows.append(row)
        full.append({**row, "retrieved": [{k: v for k, v in s.items()} for s in result["sources"]],
                     "info": {k: str(v) for k, v in result["info"].items()}})
        flag = "ok " if row["refusal_ok"] and row.get("source_hit", True) else "BAD"
        print(f"{flag} {q['id']} {q['type'][:6]:6} rel={row['best_relevance']:<5} refused={row['refused']!s:5} "
              f"hit={row.get('source_rank')} facts={row.get('fact_recall')} "
              + (f"faith={row.get('faithfulness')} relev={row.get('answer_relevancy')} judge={row.get('judge_correctness')}"
                 if use_llm else ""))

    # ---- write files ----
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (RESULTS_DIR / f"run_{stamp}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / f"run_{stamp}.json").write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    lines = [
        f"# Evaluation run {stamp}", "",
        f"- Questions: {len(rows)} · clock frozen at {EVAL_NOW:%d %b %Y %H:%M} · LLM: {mode}",
        f"- Knowledge base: {engine.stats()['articles']} articles · reranker {engine.RERANK_MODEL} · "
        f"RELEVANCE_MIN {engine.RELEVANCE_MIN}", "",
        "| Metric | Result |", "|---|---|",
        f"| Correct refusal behaviour (all) | {pct([r['refusal_ok'] for r in rows])} |",
        f"| - unanswerable questions refused | {pct([r['refusal_ok'] for r in by_type['unanswerable']])} |",
        f"| - answerable questions answered (no false refusal) | {pct([r['refusal_ok'] for r in answerable])} |",
        f"| Expected source retrieved and cited | {pct([r.get('source_hit') for r in answerable])} |",
        f"| Expected source ranked #1 | {pct([r.get('source_rank') == 1 for r in answerable])} |",
        f"| Mean fact recall | {mean([r.get('fact_recall') for r in answerable])} |",
        f"| Citations valid | {pct([r.get('citations_ok') for r in answerable])} |",
        f"| Time-sensitive: all sources inside window | {pct([r.get('recency_ok') for r in by_type['time_sensitive']])} |",
    ]
    if use_llm:
        lines += [
            f"| Faithfulness (RAGAS method, mean) | {mean([r.get('faithfulness') for r in answerable])} |",
            f"| Answer relevancy (RAGAS method, mean) | {mean([r.get('answer_relevancy') for r in answerable])} |",
            f"| LLM-judge correctness 1-5 (mean, all) | {mean([r.get('judge_correctness') for r in rows])} |",
        ]
    lines += [f"| Mean latency (s) | {mean([r['seconds'] for r in rows])} |", "", "## By type", "",
              "| Type | n | refusal ok | source hit | fact recall |", "|---|---|---|---|---|"]
    for t, rs in by_type.items():
        lines.append(f"| {t} | {len(rs)} | {pct([r['refusal_ok'] for r in rs])} | "
                     f"{pct([r.get('source_hit') for r in rs]) if t != 'unanswerable' else '-'} | "
                     f"{mean([r.get('fact_recall') for r in rs]) if t != 'unanswerable' else '-'} |")
    failures = [r for r in rows if not r["refusal_ok"] or r.get("source_hit") is False
                or (r.get("fact_recall") is not None and r["fact_recall"] < 0.5) or r.get("recency_ok") is False
                or (r.get("faithfulness") is not None and r["faithfulness"] < 0.8)]
    lines += ["", f"## Questions to look at ({len(failures)})", ""]
    for r in failures:
        why = [w for w, bad in [("wrong refusal behaviour", not r["refusal_ok"]),
                                ("expected source missing", r.get("source_hit") is False),
                                (f"fact recall {r.get('fact_recall')} (missing: {r.get('facts_missing')})",
                                 r.get("fact_recall") is not None and r["fact_recall"] < 0.5),
                                ("source outside time window", r.get("recency_ok") is False),
                                (f"faithfulness {r.get('faithfulness')}",
                                 r.get("faithfulness") is not None and r["faithfulness"] < 0.8)] if bad]
        lines.append(f"- **{r['id']}** {r['question']} - {'; '.join(why)} (best relevance {r['best_relevance']})")
    summary = "\n".join(lines) + "\n"
    (RESULTS_DIR / f"summary_{stamp}.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Saved eval/results/run_{stamp}.csv, .json and summary_{stamp}.md")


if __name__ == "__main__":
    main()
