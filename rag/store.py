"""
One place that decides WHERE the knowledge base lives - every component (ingest, agents, RAG, app) uses it.

  CHROMA_API_KEY set  -> Chroma Cloud (shared: GitHub Actions writes, Streamlit Cloud reads, laptop too)
  not set             -> local folder data_chroma/chroma (works offline, used by the evaluation)
  CHROMA_MODE=local   -> force local even when a cloud key is set (eval/run_eval.py does this)

Also pins the app clock to India time (APP_TIMEZONE, default Asia/Kolkata), because cloud servers run on UTC
and "today" / "yesterday" must mean the reader's day.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

for _name in ("CHROMA_TENANT", "CHROMA_DATABASE"):    # an empty GitHub secret arrives as "" - treat as unset
    if os.environ.get(_name) == "":
        del os.environ[_name]

# Linux / macOS servers only. Windows doesn't understand "Asia/Kolkata" in TZ and silently falls back to
# UTC, so there the PC's own time zone is kept.
if hasattr(time, "tzset"):
    os.environ.setdefault("TZ", os.getenv("APP_TIMEZONE", "Asia/Kolkata"))
    time.tzset()

LOCAL_DIR = os.getenv("CHROMA_DIR", str(PROJECT_DIR / "data_chroma" / "chroma"))
PAGE = 250                          # Chroma Cloud returns at most a few hundred records per request


def use_cloud():
    return bool(os.getenv("CHROMA_API_KEY")) and os.getenv("CHROMA_MODE", "").lower() != "local"


def describe():
    return f"Chroma Cloud ({os.getenv('CHROMA_DATABASE', 'default db')})" if use_cloud() else "local ChromaDB"


def client():
    import chromadb
    if use_cloud():
        return chromadb.CloudClient()          # reads CHROMA_API_KEY / CHROMA_TENANT / CHROMA_DATABASE
    return chromadb.PersistentClient(path=LOCAL_DIR)


def collections(c=None):
    """(articles, chunks, meta). meta holds run reports and daily briefings so the cloud app can show them."""
    c = c or client()
    space = {"hnsw:space": "cosine"}
    return (c.get_or_create_collection("articles", metadata=space),
            c.get_or_create_collection("chunks", metadata=space),
            c.get_or_create_collection("meta", metadata=space))


def get_all(collection, include, where=None):
    """collection.get() in pages - one big get() fails on Chroma Cloud once the knowledge base grows."""
    out = {"ids": [], **{k: [] for k in include}}
    offset = 0
    while True:
        batch = collection.get(include=include, where=where, limit=PAGE, offset=offset)
        out["ids"] += batch["ids"]
        for k in include:
            out[k].extend(list(batch[k]))    # embeddings come back as a numpy array
        if len(batch["ids"]) < PAGE:
            return out
        offset += PAGE


def prune(days):
    """Delete articles (and their chunks) published more than `days` ago - keeps the cloud store small."""
    articles_db, chunks_db, _ = collections()
    cutoff = int(time.time()) - days * 86400
    old = get_all(articles_db, [], where={"published_ts": {"$lt": cutoff}})["ids"]
    for i in range(0, len(old), 100):
        ids = old[i:i + 100]
        chunks_db.delete(where={"article_id": {"$in": ids}})
        articles_db.delete(ids=ids)
    return len(old)


# ---- small JSON records in the "meta" collection (run log, briefings) ------------------------------
# Chroma needs an embedding per record; these records are only fetched by id/type, so a constant is fine.

def put_record(meta_db, record_id, kind, data):
    meta_db.upsert(ids=[record_id], documents=[json.dumps(data, default=str)], embeddings=[[1.0, 0.0]],
                   metadatas=[{"kind": kind, "saved_ts": int(time.time())}])


def get_record(meta_db, record_id):
    got = meta_db.get(ids=[record_id], include=["documents"])
    return json.loads(got["documents"][0]) if got["ids"] else None


def list_records(meta_db, kind, newest=30):
    got = get_all(meta_db, ["documents", "metadatas"], where={"kind": kind})
    rows = sorted(zip(got["metadatas"], got["documents"]), key=lambda r: r[0]["saved_ts"], reverse=True)
    return [json.loads(doc) for _, doc in rows[:newest]]
