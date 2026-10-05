"""
One-time copy of the local knowledge base (data_chroma/chroma) into Chroma Cloud, so the cloud app starts
with the news you have already collected instead of an empty database.

Needs CHROMA_API_KEY, CHROMA_TENANT and CHROMA_DATABASE in .env. Then:
    venv\\Scripts\\python -m scripts.migrate_to_cloud

Safe to run twice: records are upserted by id, so nothing is duplicated.
"""

import json
import sys

import chromadb

from rag import store

BATCH = 100


def copy_collection(local, cloud, include):
    data = store.get_all(local, include)
    for i in range(0, len(data["ids"]), BATCH):
        cloud.upsert(ids=data["ids"][i:i + BATCH],
                     **{k: data[k][i:i + BATCH] for k in include})
        print(f"  {local.name}: {min(i + BATCH, len(data['ids']))}/{len(data['ids'])}", end="\r")
    print(f"  {local.name}: {len(data['ids'])} copied            ")


def main():
    if not store.use_cloud():
        sys.exit("CHROMA_API_KEY is not set in .env - nothing to copy to.")
    local = store.collections(chromadb.PersistentClient(path=store.LOCAL_DIR))
    cloud = store.collections(chromadb.CloudClient())
    print(f"Copying {store.LOCAL_DIR} -> {store.describe()}")
    for loc, cld in zip(local[:2], cloud[:2]):            # articles, chunks - keep the same embeddings
        copy_collection(loc, cld, ["documents", "metadatas", "embeddings"])

    # earlier run reports, so the app's Collection log shows the history
    logs = store.PROJECT_DIR / "data_chroma"
    for file, kind, key in (("orchestrator_log.jsonl", "run", "run_id"), ("ingest_log.jsonl", "ingest", "job_id")):
        path = logs / file
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for n, r in enumerate(rows):
                store.put_record(cloud[2], f"{kind}-{r.get(key) or r.get('time') or n}", kind, r)
            print(f"  {kind} reports: {len(rows)} copied")
    print(f"Done. Cloud now has {cloud[0].count()} articles and {cloud[1].count()} chunks.")


if __name__ == "__main__":
    main()
