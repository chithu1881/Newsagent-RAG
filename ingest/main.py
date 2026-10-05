"""
Phase 2 - News ingestion service.

n8n POSTs {"articles": [...]} to /ingest. For each article this service:
  1. skips exact duplicates (same URL already stored)
  2. fetches the full article text (falls back to the RSS snippet)
  3. skips near-duplicates (same story from another outlet in the last 3 days)
  4. extracts entities with spaCy (ORG / PERSON / GPE)      - optional
  5. writes a 2-3 line summary with Groq                     - optional
  6. chunks, embeds and stores the text in ChromaDB

Run from the Capstone1_Newsreader folder:
    venv\\Scripts\\activate
    uvicorn ingest.main:app --host 127.0.0.1 --port 8000
"""

import hashlib
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import trafilatura
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Settings
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

CHROMA_DIR = os.getenv("CHROMA_DIR", str(PROJECT_DIR / "data_chroma" / "chroma"))
LOG_FILE = Path(CHROMA_DIR).parent / "ingest_log.jsonl"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# Shared secret: n8n must send it in the "X-Ingest-Token" header (needed once the service is on a public tunnel)
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")

CHUNK_WORDS = 300          # ~400 tokens
OVERLAP_WORDS = 40         # ~50 tokens
NEAR_DUP_SIMILARITY = 0.90
NEAR_DUP_DAYS = 3
FETCH_WORKERS = 8          # full-text downloads in parallel
MIN_FULLTEXT_WORDS = 80    # shorter than this -> use the RSS snippet instead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")


# ============================================================
# 2. Load models and database (once, at startup)
# ============================================================

log.info("Loading embedding model %s ...", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# "articles": one row per article (used for duplicate checks)
# "chunks":   the text pieces the RAG chat searches
articles_db = chroma_client.get_or_create_collection("articles", metadata={"hnsw:space": "cosine"})
chunks_db = chroma_client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})

try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    log.info("spaCy loaded - entities ON")
except Exception as e:  # spaCy is optional
    nlp = None
    log.warning("spaCy not available (%s) - entities OFF", e)

groq_client = None
if GROQ_API_KEY:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    log.info("Groq key found - summaries ON (%s)", GROQ_MODEL)
else:
    log.warning("GROQ_API_KEY not set in .env - summaries OFF")

if not INGEST_TOKEN:
    log.warning("INGEST_TOKEN not set in .env - /ingest is open to anyone who can reach it")


# ============================================================
# 3. Request shape (matches the n8n "Filter & Tidy" output)
# ============================================================

class Article(BaseModel):
    title: str
    url: str
    published: str
    source: str = ""
    category: str = "unknown"
    snippet: str = ""
    keywords: list[str] = []


class IngestRequest(BaseModel):
    articles: list[Article]


# ============================================================
# 4. Helper steps
# ============================================================

def article_id(url):
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def to_timestamp(published):
    try:
        return int(datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return int(time.time())


def fetch_full_text(article):
    try:
        html = trafilatura.fetch_url(article.url)
        text = trafilatura.extract(html) if html else None
        if text and len(text.split()) >= MIN_FULLTEXT_WORDS:
            return text, "full"
    except Exception as e:
        log.debug("fetch failed %s: %s", article.url, e)
    return article.snippet or article.title, "snippet"


def embed(texts):
    return embedder.encode(texts, normalize_embeddings=True).tolist()


def is_near_duplicate(vector, published_ts):
    if articles_db.count() == 0:
        return False
    cutoff = published_ts - NEAR_DUP_DAYS * 86400
    result = articles_db.query(
        query_embeddings=[vector],
        n_results=1,
        where={"published_ts": {"$gte": cutoff}},
    )
    distances = result["distances"][0]
    return bool(distances) and (1 - distances[0]) >= NEAR_DUP_SIMILARITY


def extract_entities(text):
    if nlp is None:
        return []
    doc = nlp(text[:5000])
    found = [ent.text.strip() for ent in doc.ents if ent.label_ in ("ORG", "PERSON", "GPE")]
    return [name for name, _ in Counter(found).most_common(10)]


def summarise(title, text):
    if groq_client is None:
        return ""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=150,
            messages=[
                {"role": "system", "content": "Summarise the news article in 2-3 plain sentences. Facts only, no opinions."},
                {"role": "user", "content": f"Title: {title}\n\n{text[:4000]}"},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:  # rate limits etc. - keep going without a summary
        log.warning("summary failed for '%s': %s", title[:60], e)
        return ""


def create_chunks(text):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            break
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


# ============================================================
# 5. API
# ============================================================

app = FastAPI(title="News Analyst - Ingest")

# /ingest replies at once and processes in the background: a full batch takes several
# minutes, longer than the Cloudflare tunnel's 120 s limit. One batch runs at a time.
jobs = {}
job_runner = ThreadPoolExecutor(max_workers=1)
MAX_JOBS_KEPT = 20


@app.get("/health")
def health():
    busy = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
    return {"status": "ok", "jobs_in_progress": busy,
            "articles": articles_db.count(), "chunks": chunks_db.count()}


@app.post("/ingest", status_code=202)
def ingest(request: IngestRequest, x_ingest_token: str = Header(default="")):
    if INGEST_TOKEN and x_ingest_token != INGEST_TOKEN:
        raise HTTPException(status_code=401, detail="Wrong or missing X-Ingest-Token header")

    # Same batch already waiting? (e.g. the n8n node sent one request per item, or retried)
    batch_key = hashlib.sha1("\n".join(sorted(a.url for a in request.articles)).encode()).hexdigest()
    for j in jobs.values():
        if j["status"] in ("queued", "running") and j["batch_key"] == batch_key:
            return {"status": "already_queued", "job_id": j["job_id"], "received": len(request.articles),
                    "check_progress": f"/jobs/{j['job_id']}"}

    jobs_ahead = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + hashlib.sha1(os.urandom(8)).hexdigest()[:6]
    jobs[job_id] = {"job_id": job_id, "status": "queued", "received": len(request.articles),
                    "batch_key": batch_key, "queued_at": datetime.now(timezone.utc).isoformat()}
    finished = [k for k, j in jobs.items() if j["status"] in ("done", "failed")]
    for old in finished[:max(0, len(jobs) - MAX_JOBS_KEPT)]:
        jobs.pop(old)

    job_runner.submit(run_job, job_id, request.articles)
    return {"status": "accepted", "job_id": job_id, "received": len(request.articles),
            "jobs_ahead": jobs_ahead, "check_progress": f"/jobs/{job_id}"}


@app.get("/jobs/latest")
def latest_job():
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs since the service started")
    return jobs[list(jobs)[-1]]


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Unknown job id (the service may have restarted)")
    return jobs[job_id]


def run_job(job_id, articles):
    jobs[job_id]["status"] = "running"
    try:
        result = process_batch(articles, job_id)
        jobs[job_id].update(status="done", **result)
    except Exception as e:
        log.exception("job %s failed", job_id)
        jobs[job_id].update(status="failed", error=str(e))


def process_batch(articles, job_id):
    started = time.time()
    report = Counter(received=len(articles))
    per_category = Counter()

    # Step 1: exact duplicates (same URL, or repeated within this batch)
    new_articles, seen = [], set()
    for a in articles:
        aid = article_id(a.url)
        if aid in seen or articles_db.get(ids=[aid])["ids"]:
            report["skipped_duplicate"] += 1
            continue
        seen.add(aid)
        new_articles.append(a)

    # Step 2: download full text in parallel (the slow part)
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        texts = list(pool.map(fetch_full_text, new_articles))

    for article, (text, text_type) in zip(new_articles, texts):
        try:
            aid = article_id(article.url)
            published_ts = to_timestamp(article.published)

            # Step 3: near-duplicates (title + first paragraph)
            lead = article.title + ". " + " ".join(text.split()[:60])
            lead_vector = embed([lead])[0]
            if is_near_duplicate(lead_vector, published_ts):
                report["skipped_near_duplicate"] += 1
                continue

            # Steps 4-5: entities + summary
            entities = extract_entities(text)
            summary = summarise(article.title, text)

            meta = {
                "article_id": aid,
                "title": article.title,
                "url": article.url,
                "source": article.source,
                "category": article.category,
                "published": article.published,
                "published_ts": published_ts,
                "entities": ", ".join(entities),
                "keywords": ", ".join(article.keywords),
                "summary": summary,
                "text_type": text_type,
            }

            # Step 6: chunk + embed + store (title on every chunk helps search)
            chunks = create_chunks(text)
            chunk_texts = [f"{article.title}\n{c}" for c in chunks]
            chunks_db.add(
                ids=[f"{aid}-{i}" for i in range(len(chunks))],
                documents=chunk_texts,
                embeddings=embed(chunk_texts),
                metadatas=[{**meta, "chunk_index": i} for i in range(len(chunks))],
            )
            articles_db.add(ids=[aid], documents=[lead], embeddings=[lead_vector], metadatas=[meta])

            report["stored"] += 1
            report["chunks_added"] += len(chunks)
            report[f"text_{text_type}"] += 1
            per_category[article.category] += 1
        except Exception as e:
            report["failed"] += 1
            log.exception("failed to store %s: %s", article.url, e)

    result = {
        "job_id": job_id,
        "time": datetime.now(timezone.utc).isoformat(),
        **{k: report.get(k, 0) for k in ("received", "stored", "skipped_duplicate",
                                           "skipped_near_duplicate", "failed", "chunks_added",
                                           "text_full", "text_snippet")},
        "per_category": dict(per_category),
        "seconds": round(time.time() - started, 1),
        "totals": {"articles": articles_db.count(), "chunks": chunks_db.count()},
    }

    # Daily collection report: one line per n8n run
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    log.info("ingest report: %s", result)
    return result
