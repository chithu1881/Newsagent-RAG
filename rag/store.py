"""
Where the knowledge base lives - a ChromaDB folder - and how the cloud app gets the latest copy.

  Laptop            data_chroma/chroma, written by the ingest service / orchestrator.
  GitHub Actions    downloads the current database, adds new news, and saves it back as kb.zip on the
                    repo's "kb-data" branch (scripts/kb_sync.py). No database account or keys needed.
  Streamlit Cloud   downloads kb.zip from that branch at start and every 30 minutes (sync_from_github).

KB_SOURCE=local|github overrides the automatic choice (github on Streamlit Cloud, local everywhere else).
Also pins the clock to India time on Linux servers (APP_TIMEZONE), which run on UTC.
"""

import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

# Linux / macOS servers only. Windows doesn't understand "Asia/Kolkata" in TZ and silently falls back to
# UTC, so there the PC's own time zone is kept.
if hasattr(time, "tzset"):
    os.environ.setdefault("TZ", os.getenv("APP_TIMEZONE", "Asia/Kolkata"))
    time.tzset()

LOCAL_DIR = os.getenv("CHROMA_DIR", str(PROJECT_DIR / "data_chroma" / "chroma"))
KB_URL = os.getenv("KB_URL", "https://raw.githubusercontent.com/chithu1881/Newsagent-RAG/kb-data/kb.zip")
DOWNLOAD_ROOT = PROJECT_DIR / "data_chroma" / "from_github"
PAGE = 250                          # rows per get() call, so big collections are read in pages

_active = {"dir": LOCAL_DIR, "etag": None, "checked": 0.0}


def source():
    wanted = os.getenv("KB_SOURCE", "").lower()
    if wanted in ("local", "github"):
        return wanted
    return "github" if Path("/mount/src").exists() else "local"     # /mount/src = Streamlit Community Cloud


def describe():
    return "GitHub (kb-data branch)" if source() == "github" else "local ChromaDB"


def sync_from_github(every_seconds=1800):
    """Cloud app: fetch the latest kb.zip if it changed. Returns True when a new copy was installed."""
    if source() != "github" or time.time() - _active["checked"] < every_seconds:
        return False
    import requests
    _active["checked"] = time.time()
    headers = {"If-None-Match": _active["etag"]} if _active["etag"] else {}
    try:
        r = requests.get(KB_URL, headers=headers, timeout=120)
    except requests.RequestException:
        return False                        # keep using the copy we have
    if r.status_code != 200:                # 304 = unchanged, 404 = no data published yet
        return False
    target = DOWNLOAD_ROOT / str(int(time.time()))
    target.mkdir(parents=True)
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(target)
    previous = _active["dir"]
    _active.update(dir=str(target), etag=r.headers.get("ETag"))
    # remove older copies, but keep the previous one: open connections may still be reading it
    for d in DOWNLOAD_ROOT.iterdir():
        if str(d) not in (str(target), previous):
            shutil.rmtree(d, ignore_errors=True)
    return True


def client():
    import chromadb
    if source() == "github" and _active["dir"] == LOCAL_DIR:
        sync_from_github(every_seconds=0)   # first use on the cloud: download before opening
    Path(_active["dir"]).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=_active["dir"])


def collections(c=None):
    """(articles, chunks, meta). meta holds run reports and daily briefings."""
    c = c or client()
    space = {"hnsw:space": "cosine"}
    return (c.get_or_create_collection("articles", metadata=space),
            c.get_or_create_collection("chunks", metadata=space),
            c.get_or_create_collection("meta", metadata=space))


def get_all(collection, include, where=None):
    """collection.get() in pages."""
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
    """Delete articles (and their chunks) published more than `days` ago - keeps kb.zip small."""
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
