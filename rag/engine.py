"""
Phase 3 - RAG query engine.

answer(question) runs five steps:
  1. understand  - infer a time range ("this week", "yesterday", "last 3 days") and a category
  2. retrieve    - hybrid search: Chroma (semantic) + BM25 (keyword), fused with Reciprocal Rank Fusion
  3. rerank      - cross-encoder relevance, plus a boost for recent news and the matching category
  4. generate    - LLM answers ONLY from the retrieved chunks, citing them as [1], [2] ...
  5. cite        - returns the cited sources with links and dates; if nothing relevant was found
                   the answer is "I don't have news on that." (no guessing)

Quick test from the Capstone1_Newsreader folder:
    venv\\Scripts\\python -m rag.engine "What did the RBI announce this week?"
"""

import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from rag import llm

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

CHROMA_DIR = os.getenv("CHROMA_DIR", str(PROJECT_DIR / "data_chroma" / "chroma"))
EMBED_MODEL = "BAAI/bge-small-en-v1.5"                      # must match ingest/main.py
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge query instruction
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

CANDIDATES = 40            # chunks taken from each search before fusion
RERANK_TOP = 20            # fused chunks sent to the cross-encoder
RERANK_MAX_TOKENS = 512    # 256 is faster but scored clearly worse in testing
FINAL_CHUNKS = 8           # chunks given to the LLM
MAX_CHUNKS_PER_ARTICLE = 2
RELEVANCE_MIN = float(os.getenv("RELEVANCE_MIN", "0.10"))   # below this best score -> "no news"
RECENCY_HALF_LIFE_DAYS = 3.0
W_RELEVANCE, W_RECENCY, W_CATEGORY = 0.75, 0.15, 0.10

NO_NEWS = "I don't have news on that."
CATEGORIES = ("tech", "finance", "politics")


# ============================================================
# 1. Understand the question (rules - fast, free, testable)
# ============================================================

CATEGORY_WORDS = {
    "tech": ["ai", "artificial intelligence", "tech", "technology", "startup", "software", "chip",
             "semiconductor", "smartphone", "app", "cyber", "cloud", "telecom", "5g", "openai",
             "google", "apple", "microsoft", "meta", "nvidia", "infosys", "tcs", "wipro", "gadget"],
    "finance": ["rbi", "repo", "rate cut", "rate hike", "inflation", "sensex", "nifty", "market",
                "stock", "share", "ipo", "sebi", "rupee", "bank", "earnings", "results", "gdp",
                "fii", "mutual fund", "bond", "economy", "gst", "tax", "budget", "investor"],
    "politics": ["parliament", "lok sabha", "rajya sabha", "minister", "election", "bjp", "congress",
                 "government", "cabinet", "supreme court", "high court", "bill", "opposition",
                 "chief minister", "modi", "policy", "party", "poll", "vote", "governor"],
}


def _has_word(text, word):
    return re.search(rf"\b{re.escape(word)}s?\b", text) is not None


def understand(question, now=None):
    """Return {date_from, date_to, categories, time_label} inferred from the question text."""
    now = now or datetime.now()
    q = question.lower()
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    date_from = date_to = None
    label = ""

    m = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+(hour|day|week|month)s?\b", q)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        span = {"hour": timedelta(hours=n), "day": timedelta(days=n),
                "week": timedelta(weeks=n), "month": timedelta(days=30 * n)}[unit]
        date_from, label = now - span, f"last {n} {unit}{'s' if n > 1 else ''}"
    elif re.search(r"\btoday\b|\btoday's\b|\bthis morning\b|\btonight\b", q):
        date_from, label = start_of_today, "today"
    elif re.search(r"\byesterday\b", q):
        date_from, date_to, label = start_of_today - timedelta(days=1), start_of_today, "yesterday"
    elif re.search(r"\b(this|past|last)\s+week\b|\bweek's\b", q):
        date_from, label = now - timedelta(days=7), "last 7 days"
    elif re.search(r"\b(this|past|last)\s+month\b", q):
        date_from, label = now - timedelta(days=30), "last 30 days"

    scores = {c: sum(_has_word(q, w) for w in words) for c, words in CATEGORY_WORDS.items()}
    best = max(scores.values())
    categories = [c for c, s in scores.items() if s == best] if best > 0 else []

    return {"date_from": date_from, "date_to": date_to, "categories": categories, "time_label": label,
            "topic": topic_of(question)}


TIME_PHRASES = re.compile(
    r"\b(?:in\s+|over\s+|during\s+)?(?:the\s+)?(?:(?:last|past|previous)\s+\d+\s+(?:hour|day|week|month)s?"
    r"|(?:this|past|last)\s+(?:week|month)|today'?s?|yesterday|this morning|tonight|lately|recently)\b", re.I)
FILLER = re.compile(r"\b(?:any|anything|latest|recent|news|updates?|headlines?|stories|story|tell me|"
                    r"about|please|what's new|what is new)\b", re.I)


def topic_of(question):
    """Question without time phrases and filler - what the cross-encoder should judge relevance on."""
    t = FILLER.sub(" ", TIME_PHRASES.sub(" ", question))
    t = re.sub(r"\s+", " ", t).strip(" ?.!,")
    return t or question


# ============================================================
# 2. Load models + data (cached - loaded once per process)
# ============================================================

@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_TOKENS)


@lru_cache(maxsize=1)
def _collections():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return (client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"}),
            client.get_or_create_collection("articles", metadata={"hnsw:space": "cosine"}))


def chunks_db():
    return _collections()[0]


def articles_db():
    return _collections()[1]


_bm25_cache = {"count": -1}


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_corpus():
    """All chunks + a BM25 index; rebuilt only when the chunk count changes (new ingest)."""
    db = chunks_db()
    count = db.count()
    if _bm25_cache["count"] != count:
        from rank_bm25 import BM25Okapi
        data = db.get(include=["documents", "metadatas"])
        _bm25_cache.update(
            count=count, ids=data["ids"], docs=data["documents"], metas=data["metadatas"],
            index=BM25Okapi([_tokens(d) for d in data["documents"]]) if count else None,
        )
    return _bm25_cache


def stats():
    db = articles_db()
    metas = db.get(include=["metadatas"])["metadatas"] if db.count() else []
    per_cat = defaultdict(int)
    newest = 0
    for m in metas:
        per_cat[m.get("category", "unknown")] += 1
        newest = max(newest, m.get("published_ts", 0))
    return {"articles": len(metas), "chunks": chunks_db().count(), "per_category": dict(per_cat),
            "newest": datetime.fromtimestamp(newest) if newest else None}


# ============================================================
# 3. Retrieve (hybrid) and rerank
# ============================================================

def _where(categories, ts_from, ts_to):
    conditions = []
    if categories:
        conditions.append({"category": {"$in": list(categories)}})
    if ts_from:
        conditions.append({"published_ts": {"$gte": ts_from}})
    if ts_to:
        conditions.append({"published_ts": {"$lt": ts_to}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _passes(meta, categories, ts_from, ts_to):
    ts = meta.get("published_ts", 0)
    return ((not categories or meta.get("category") in categories)
            and (not ts_from or ts >= ts_from) and (not ts_to or ts < ts_to))


def retrieve(query, categories=None, ts_from=None, ts_to=None):
    """Hybrid search -> list of {id, text, meta} fused with Reciprocal Rank Fusion."""
    db = chunks_db()
    if db.count() == 0:
        return []
    fused = defaultdict(float)
    pool = {}

    # Semantic search (Chroma)
    vector = _embedder().encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    try:
        res = db.query(query_embeddings=vector, n_results=min(CANDIDATES, db.count()),
                       where=_where(categories, ts_from, ts_to), include=["documents", "metadatas"])
        for rank, (cid, doc, meta) in enumerate(zip(res["ids"][0], res["documents"][0], res["metadatas"][0])):
            fused[cid] += 1 / (60 + rank)
            pool[cid] = {"id": cid, "text": doc, "meta": meta}
    except Exception:  # Chroma raises if the filter leaves fewer docs than n_results on some versions
        pass

    # Keyword search (BM25) over the same filtered set
    corpus = _bm25_corpus()
    if corpus["index"] is not None:
        scores = corpus["index"].get_scores(_tokens(query))
        ranked = sorted((i for i, s in enumerate(scores)
                         if s > 0 and _passes(corpus["metas"][i], categories, ts_from, ts_to)),
                        key=lambda i: scores[i], reverse=True)[:CANDIDATES]
        for rank, i in enumerate(ranked):
            cid = corpus["ids"][i]
            fused[cid] += 1 / (60 + rank)
            pool.setdefault(cid, {"id": cid, "text": corpus["docs"][i], "meta": corpus["metas"][i]})

    order = sorted(fused, key=fused.get, reverse=True)
    return [pool[cid] for cid in order]


def rerank(query, candidates, boost_categories=(), now_ts=None):
    """Score = relevance (cross-encoder) + recency + category match. Returns best-first."""
    if not candidates:
        return []
    now_ts = now_ts or time.time()
    candidates = candidates[:RERANK_TOP]
    raw = _reranker().predict([(query, c["text"]) for c in candidates])
    for c, r in zip(candidates, raw):
        r = float(r)
        relevance = r if 0.0 <= r <= 1.0 else 1 / (1 + math.exp(-r))  # some models return logits
        age_days = max(0.0, (now_ts - c["meta"].get("published_ts", now_ts)) / 86400)
        recency = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
        category = 1.0 if c["meta"].get("category") in boost_categories else 0.0
        c["relevance"] = relevance
        c["score"] = W_RELEVANCE * relevance + W_RECENCY * recency + W_CATEGORY * category
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def select_context(ranked):
    """Best chunks, at most MAX_CHUNKS_PER_ARTICLE per article, grouped into numbered sources."""
    picked, per_article = [], defaultdict(int)
    for c in ranked:
        aid = c["meta"].get("article_id", c["id"])
        if c["relevance"] < RELEVANCE_MIN / 2 or per_article[aid] >= MAX_CHUNKS_PER_ARTICLE:
            continue
        per_article[aid] += 1
        picked.append(c)
        if len(picked) >= FINAL_CHUNKS:
            break

    sources, by_article = [], {}
    for c in picked:
        m = c["meta"]
        aid = m.get("article_id", c["id"])
        if aid not in by_article:
            by_article[aid] = {
                "n": len(sources) + 1, "title": m.get("title", ""), "source": m.get("source", ""),
                "url": m.get("url", ""), "category": m.get("category", ""),
                "published_ts": m.get("published_ts", 0), "date": fmt_date(m.get("published_ts", 0)),
                "summary": m.get("summary", ""), "score": round(c["score"], 3), "chunks": [],
            }
            sources.append(by_article[aid])
        by_article[aid]["chunks"].append(c["text"])
    return sources


def fmt_date(ts):
    return datetime.fromtimestamp(ts).strftime("%d %b %Y, %H:%M") if ts else "date unknown"


# ============================================================
# 4. Generate with citations
# ============================================================

SYSTEM_PROMPT = """You are a news analyst. Today is {today}.
Answer the user's question using ONLY the numbered news sources provided. Rules:
- Every factual sentence must end with its citation(s) like [1] or [2][3]. Only cite source numbers that exist.
- Be date-aware: say when things happened (e.g. "On 2 Oct, ..."), prefer the most recent reports, and point out
  if sources disagree or if one is an update of another.
- Do not use outside knowledge, do not speculate, and do not invent numbers.
- If the sources do not contain the answer, reply with exactly: {no_news}
- If they answer only part of the question, answer that part and say plainly which part is not covered.
- Keep it short: a direct 1-2 sentence answer first, then up to 5 bullet points of supporting detail."""


def build_prompt(question, sources, time_label):
    blocks = []
    for s in sources:
        body = "\n...\n".join(s["chunks"])
        blocks.append(f"[{s['n']}] {s['title']}\nSource: {s['source']} | Published: {s['date']} | "
                      f"Category: {s['category']}\n{body}")
    scope = f" (time range asked: {time_label})" if time_label else ""
    return f"Question{scope}: {question}\n\nSources:\n\n" + "\n\n---\n\n".join(blocks)


def extractive_answer(sources):
    """Used when no LLM key is configured: show the best-matching articles instead of an answer."""
    lines = ["*No LLM key configured - showing the most relevant articles instead of a written answer.*", ""]
    for s in sources[:5]:
        body = " ".join(c.split("\n", 1)[-1] for c in s["chunks"])   # chunks start with the title line
        text = s["summary"] or " ".join(body.split()[:50]) + " ..."
        lines.append(f"- **{s['title']}** ({s['date']}): {text} [{s['n']}]")
    return "\n".join(lines)


def answer(question, categories=None, date_from=None, date_to=None, history=None):
    """
    question    - user text
    categories  - UI filter (list of tech/finance/politics); overrides inference
    date_from/to- UI filter (datetime); overrides inference
    history     - previous [(question, answer)] pairs, used to make short follow-ups searchable
    """
    started = time.time()
    now = datetime.now()
    parsed = understand(question, now)

    # UI filters win; otherwise use what was inferred. Inferred categories only *boost*, they don't filter,
    # because many stories span two domains (e.g. an AI bill in Parliament).
    hard_categories = list(categories) if categories else None
    d_from = date_from or parsed["date_from"]
    d_to = date_to or parsed["date_to"]
    time_label = parsed["time_label"] if not (date_from or date_to) else (
        f"{date_from:%d %b} to {date_to:%d %b}" if date_from and date_to else "custom range")

    # Short follow-ups ("and how did markets react?") need the previous question for retrieval
    search_query = parsed["topic"]
    if history and len(question.split()) < 10:
        search_query = f"{topic_of(history[-1][0])} {search_query}"

    ts_from = int(d_from.timestamp()) if d_from else None
    ts_to = int(d_to.timestamp()) if d_to else None
    # Compound questions ("what did X announce and how did markets react") - search each part, then merge
    parts = [p.strip() for p in re.split(r"\band\s+(?=how|what|why|who|when|where|did|was|were|is|are)\b",
                                         search_query, flags=re.I) if len(p.split()) >= 2]
    seen, candidates = set(), []
    for part in ([search_query] + parts if len(parts) > 1 else [search_query]):
        for c in retrieve(part, hard_categories, ts_from, ts_to):
            if c["id"] not in seen:
                seen.add(c["id"])
                candidates.append(c)
    ranked = rerank(search_query, candidates, boost_categories=parsed["categories"], now_ts=now.timestamp())

    info = {"time_label": time_label or "any time", "date_from": d_from, "date_to": d_to,
            "categories_filter": hard_categories or [], "categories_inferred": parsed["categories"],
            "candidates": len(candidates), "best_relevance": round(ranked[0]["relevance"], 3) if ranked else 0.0,
            "llm": llm.describe()}

    if not ranked or ranked[0]["relevance"] < RELEVANCE_MIN:
        scope = f" for {time_label}" if time_label else ""
        info["seconds"] = round(time.time() - started, 1)
        return {"answer": NO_NEWS[:-1] + scope + ".", "sources": [], "info": info}

    sources = select_context(ranked)
    if llm.available():
        system = SYSTEM_PROMPT.format(today=now.strftime("%A, %d %B %Y"), no_news=NO_NEWS)
        text = llm.chat(system, build_prompt(question, sources, time_label))
        if not text:
            text = extractive_answer(sources) + "\n\n*(The LLM call failed, so this is the fallback view.)*"
    else:
        text = extractive_answer(sources)

    if text.strip().startswith(NO_NEWS[:-1]):
        used = []
    else:
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        used = [s for s in sources if s["n"] in cited] or sources

    info["seconds"] = round(time.time() - started, 1)
    return {"answer": text, "sources": used, "info": info}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What did the RBI announce this week and how did markets react?"
    result = answer(q)
    print("\nQ:", q)
    print("Filters:", {k: v for k, v in result["info"].items()})
    print("\n" + result["answer"] + "\n")
    for s in result["sources"]:
        print(f"[{s['n']}] {s['title']} - {s['source']}, {s['date']}\n    {s['url']}")
