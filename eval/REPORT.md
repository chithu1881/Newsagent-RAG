# Evaluation Report: Multi-Agent AI News Analyst

*Phase 5 · draft of 5 Oct 2026. Sections marked **⏳ pending** are filled in after the first run with an LLM key.*

## 1. What was tested and how

**Question set** (`eval/questions.csv`): 25 questions written against the knowledge-base snapshot of 5 Oct 2026 (359 articles: 121 tech, 180 finance, 58 politics).

| Type | n | What it tests |
|---|---|---|
| Answerable | 15 | 5 per domain; each has an expected source article, key facts and a human reference answer |
| Time-sensitive | 5 | "today", "yesterday", "this week", "last 3 days", plus one trap: *"Has the RBI announced its policy decision this week?"* The correct answer is "not yet". |
| Unanswerable | 5 | 2 out of domain (IPL, weather) and 3 **near-misses** whose topic is in the news but whose answer is not (Apple WWDC, Tesla Q3 deliveries, India's Q1 GDP figure) |

**Repeatable runs:** every question is answered with the clock frozen at 5 Oct 2026 23:00. "Today" and "this week" always mean the same dates, and articles collected after that time are ignored, so re-runs can be compared while new news keeps arriving.

**Metrics** (`eval/run_eval.py`):

| Metric | How | Plan criterion |
|---|---|---|
| Refusal behaviour | Unanswerable questions must give "I don't have news on that."; the others must not | correct refusal |
| Source hit / rank | The expected article is among the cited sources, and at what position | citation correctness |
| Fact recall | Share of expected key facts present in the answer | accuracy (automatic proxy) |
| Citations valid | Every `[n]` points to a returned source | citation correctness |
| Recency | Time-sensitive questions: every source falls inside the asked window | recency |
| Faithfulness, answer relevancy | RAGAS method (claims checked against the context; generated questions compared with the asked one by cosine similarity) | RAGAS |
| Judge correctness 1–5 and manual 1–5 | LLM grade against the reference answer, plus an empty column for my own score | accuracy |

RAGAS itself could not be installed: its dependency `scikit-network` has no build for Python 3.14. The two RAGAS metrics are therefore implemented in `run_eval.py` with the same method, using the project's own LLM and the bge embedder.

## 2. Results

### Run 1 (baseline) and run 2 (after fix), no LLM key (retrieval / extractive mode)

| Metric | Run 1 | Run 2 |
|---|---|---|
| Unanswerable questions refused | 2/5 | 2/5 |
| Answerable questions answered (no false refusal) | 19/20 | **20/20** |
| Expected source retrieved and cited | 19/20 | **20/20** |
| Expected source ranked #1 | 16/20 | 17/20 |
| Mean fact recall | 0.89 | 0.89 |
| Citations valid | 19/20 | **20/20** |
| Time-sensitive: sources inside window | 4/5 | **5/5** |
| Mean latency | 14.7 s | 16.6 s |

Files: `eval/results/summary_20261005-2256.md` (run 1) and `summary_20261005-2305.md` (run 2).

### Run 3: reranker comparison (no LLM key)

Same 25 questions, with only the cross-encoder swapped. The smaller model was tried for the cloud deployment (1 GB memory limit) and turned out better on every measure:

| Metric | bge-reranker-base (run 2) | ms-marco-MiniLM-L-6-v2 (run 3) |
|---|---|---|
| Unanswerable questions refused | 2/5 | **3/5** (now also catches Apple WWDC) |
| False refusals | 0/20 | 0/20 |
| Expected source ranked #1 | 17/20 | **19/20** |
| Time windows respected | 5/5 | 5/5 |
| Mean latency | 16.6 s | **5.4 s** |
| Model size | ~1.1 GB | ~90 MB |

**Decision:** MiniLM is now the default reranker. Run 4 (`summary_20261005-2346.md`), taken after fixes 3 and 4 below, reproduced run 3 exactly, at 3.0 s per question.

### Run 5 with LLM ⏳ pending

Faithfulness, answer relevancy, judge correctness, manual scores, and the refusal rate on the near-miss questions.

## 3. Failure patterns

**F1. Digest questions were wrongly refused (Q19, fixed).** *"What were the main AI policy stories yesterday?"* was refused although four AI-policy stories from 4 Oct were stored. Cause: the query cleaner removed "stories" but left "What were the main AI policy". The cross-encoder scored that broken fragment 0.05, below the 0.10 relevance gate. The clean topic "AI policy" scores 0.68.
*Fix:* for digest-style questions (main/top/key … stories/news/headlines), also strip the question scaffolding. The best match rose from 0.051 to 0.534, the question is now answered from the right dates, and the regression checks (`eval/smoke_test.py`, 22/22) still pass.

**F2. Near-miss unanswerable questions pass the retrieval gate (Q23–Q25, open).** The gate refuses only when nothing in the knowledge base is about the topic. For in-domain questions whose specific answer isn't stored, related articles score as relevant: Tesla Q3 deliveries retrieved Tesla/robotaxi stories (0.84), and GDP growth retrieved rupee and IPO stories (0.19). The cross-encoder measures *topic relevance*, not *answerability*.
*Design response:* this is why the engine has two layers. The second layer is the LLM prompt rule: *"If the sources do not contain the answer, reply with exactly: I don't have news on that."* In run 1 and run 2 there was no LLM, so the second layer never ran. Run 3 will show whether it catches these.
*Rejected tweak:* raising `RELEVANCE_MIN` to 0.30 would refuse Q23 (0.29) and Q25 (0.19) with no false refusals in this set. But it would not catch Q24 (0.84), it overfits 25 questions, and broad digest questions legitimately score lower.

**F3. Facts outside the first 50 words (Q11, Q19, extractive mode only).** The Supreme Court judges' names and "super intelligence" sit later in the article than the 50-word preview. The right article is retrieved (source hit), but fact recall is low. This should disappear once the LLM writes the answer from the full chunks. To verify in run 3.

**F4. Latency (fixed).** The bge-reranker-base cross-encoder on CPU took 7–17 s per question. Switching to MiniLM (run 3) brought it to 3–5 s and improved quality too.

**F5. Domain digests after the reranker swap (fixed, found by the smoke test).** *"What are the main political stories this week?"* reduces to the single word "political". MiniLM scores one generic word below the relevance gate, so the question was refused. A relevance score is meaningless for "give me the main stories in X".
*Fix:* digest questions whose remaining topic only names a domain now filter to that domain and time window and return the newest stories, without the relevance gate. The smoke test is back to 22/22. This question is not in the 25-question set, which is a reminder that the regression checks cover cases the eval set doesn't.

**F6. Wrong clock on Windows (fixed).** To make "today" mean India time on cloud servers, which run on UTC, the code set `TZ=Asia/Kolkata`. Windows doesn't understand that name and silently switched to UTC. A collection run was stamped 18:10 instead of 23:40, and the morning-briefing check fired at night. The zone is now set only on Linux and macOS servers.

## 4. Fixes made during evaluation

| # | Change | Evidence |
|---|---|---|
| 1 | Digest-question scaffolding stripped before reranking (`rag/engine.py: topic_of`) | Q19: refused → answered; time window 4/5 → 5/5; no regressions |
| 2 | Frozen evaluation clock: `answer(..., now=)` ignores later articles, and the eval reads the frozen `eval/kb_snapshot` | Re-runs are repeatable while the live store changes |
| 3 | Reranker bge-reranker-base → ms-marco-MiniLM-L-6-v2 | Run 3: refusals 2/5 → 3/5, #1 rank 17/20 → 19/20, 16.6 s → 5.4 s |
| 4 | Domain-digest questions bypass the relevance gate (F5); clock fix (F6) | Smoke 22/22; run 4 identical to run 3 |
| *earlier* | Time phrases and filler stripped from the reranker query | "AI startup funding … last 5 days": 0.09 → 0.89 |
| *earlier* | Reranker max length back to 512 tokens (256 hurt scores badly) | RBI question: 0.17 → 0.71 |
| *earlier* | Compound questions searched part by part | "… and how did markets react" sources now retrieved |

## 5. Next steps

1. Add an LLM key and run `python -m eval.run_eval` for run 5; fill in sections 2 and 3.
2. Manually score the run-5 answers (`manual_accuracy_1to5` column in the CSV) and compare them with the LLM judge.
3. If near-miss refusals stay weak (Tesla Q24 and GDP Q25 score 0.85–0.93 relevance), add an explicit answerability check: a short LLM yes/no call before answering.
4. Add the F5 digest question to `questions.csv` so the full eval covers it.
