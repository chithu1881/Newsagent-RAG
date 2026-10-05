"""
Orchestrator - a LangGraph graph that runs the whole collection pipeline.

        START
   ┌──────┼──────────┐
   ▼      ▼          ▼
 tech   finance   politics      fetcher agents run in parallel
   └──────┼──────────┘
          ▼
   processing_agent             clean, dedupe, summarise, entities, chunk, embed -> knowledge base
          ▼                     (via the ingest web service, or in-process with --inline)
    housekeeping                delete news older than RETENTION_DAYS; write the morning briefing
          ▼
       log_run                  run report -> data_chroma/orchestrator_log.jsonl + knowledge base
          ▼
         END

Failures: each feed is retried 3x inside its agent; a failed agent is logged and the run continues
with the others; the processing step is retried 3x by LangGraph (e.g. ingest service unreachable).

    venv\\Scripts\\python -m agents.orchestrator              # run once (needs start_ingest.bat running)
    venv\\Scripts\\python -m agents.orchestrator --inline     # run once, process in this process (no service)
    venv\\Scripts\\python -m agents.orchestrator --schedule   # every FETCH_EVERY_HOURS
    venv\\Scripts\\python -m agents.orchestrator --briefing   # only (re)write today's briefing

In the cloud, .github/workflows/collect.yml runs "--inline" every 3 hours on GitHub Actions and writes to
Chroma Cloud (see rag/store.py). The n8n workflow (n8n/01_news_fetcher.json) is the visual alternative;
it posts to the same /ingest endpoint.
"""

import argparse
import json
import logging
import operator
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from agents.fetchers import FetcherAgent
from rag import store

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

INGEST_URL = os.getenv("INGEST_URL", "http://127.0.0.1:8000").rstrip("/")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")
FETCH_EVERY_HOURS = int(os.getenv("FETCH_EVERY_HOURS", "3"))
BRIEFING_HOUR = int(os.getenv("BRIEFING_HOUR", "7"))
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "30"))
JOB_TIMEOUT_MIN = 30
RUN_LOG = PROJECT_DIR / "data_chroma" / "orchestrator_log.jsonl"
INLINE = False                                    # set by --inline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestrator")


class RunState(TypedDict, total=False):
    run_id: str
    articles: Annotated[list, operator.add]       # each agent appends its articles
    agent_reports: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    ingest: dict
    housekeeping: dict


# ---- nodes -------------------------------------------------------------------

def make_fetcher_node(category):
    def node(state: RunState):
        try:
            articles, report = FetcherAgent(category).run()
            return {"articles": articles, "agent_reports": [report]}
        except Exception as e:  # one agent failing must not stop the others
            log.exception("%s agent crashed", category)
            return {"agent_reports": [{"agent": category, "kept": 0}], "errors": [f"{category} agent: {e}"]}
    node.__name__ = f"{category}_agent"
    return node


def processing_agent(state: RunState):
    """Hands the merged batch to the processing pipeline and returns its report."""
    articles, seen = [], set()
    for a in state.get("articles", []):          # the same story can come from two agents
        if a["url"] not in seen:
            seen.add(a["url"])
            articles.append(a)
    if not articles:
        return {"ingest": {"status": "skipped", "reason": "no articles kept by the agents"}}

    if INLINE:
        from ingest.main import process_inline   # loads the models here, only in inline mode
        return {"ingest": process_inline(articles)}

    headers = {"X-Ingest-Token": INGEST_TOKEN} if INGEST_TOKEN else {}
    r = requests.post(f"{INGEST_URL}/ingest", json={"articles": articles}, headers=headers, timeout=60)
    r.raise_for_status()                         # LangGraph retries this node on failure
    job_id = r.json()["job_id"]
    log.info("ingest accepted %d articles as job %s", len(articles), job_id)

    deadline = time.time() + JOB_TIMEOUT_MIN * 60
    while time.time() < deadline:
        time.sleep(5)
        job = requests.get(f"{INGEST_URL}/jobs/{job_id}", timeout=30).json()
        if job.get("status") in ("done", "failed"):
            return {"ingest": job}
    return {"ingest": {"status": "timeout", "job_id": job_id}}


def housekeeping(state: RunState):
    done, errors = {}, []
    try:
        done["pruned_articles"] = store.prune(RETENTION_DAYS)
    except Exception as e:
        errors.append(f"prune: {e}")
    try:
        from rag.briefing import get_briefing, needs_morning_briefing
        if needs_morning_briefing(BRIEFING_HOUR):
            get_briefing(force=True)
            done["briefing"] = "written"
    except Exception as e:
        errors.append(f"briefing: {e}")
    return {"housekeeping": done, "errors": errors}


def log_run(state: RunState):
    ingest = state.get("ingest", {})
    entry = {
        "run_id": state["run_id"],
        "finished": datetime.now(timezone.utc).isoformat(),
        "where": "GitHub Actions" if os.getenv("GITHUB_ACTIONS") else "local",
        "agents": {r["agent"]: {k: v for k, v in r.items() if k != "agent"} for r in state.get("agent_reports", [])},
        "sent_to_ingest": len({a["url"] for a in state.get("articles", [])}),
        "ingest": {k: ingest.get(k) for k in ("status", "stored", "skipped_duplicate", "skipped_near_duplicate",
                                               "failed", "chunks_added", "per_category", "seconds", "error")
                   if k in ingest},
        "housekeeping": state.get("housekeeping", {}),
        "errors": state.get("errors", []),
    }
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    try:
        store.put_record(store.collections()[2], f"run-{state['run_id']}", "run", entry)
    except Exception as e:
        log.warning("could not save run report to the knowledge base: %s", e)
    log.info("run report: %s", json.dumps(entry, indent=1))
    return {}


# ---- graph ---------------------------------------------------------------------

def build_graph():
    g = StateGraph(RunState)
    agents = []
    for category in ("tech", "finance", "politics"):
        name = f"{category}_agent"
        g.add_node(name, make_fetcher_node(category))
        g.add_edge(START, name)
        agents.append(name)
    g.add_node("processing_agent", processing_agent,
               retry_policy=RetryPolicy(max_attempts=3, initial_interval=10, backoff_factor=3))
    g.add_node("housekeeping", housekeeping)
    g.add_node("log_run", log_run)
    g.add_edge(agents, "processing_agent")       # waits for all three agents
    g.add_edge("processing_agent", "housekeeping")
    g.add_edge("housekeeping", "log_run")
    g.add_edge("log_run", END)
    return g.compile()


def run_once():
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log.info("=== collection run %s -> %s ===", run_id, store.describe())
    try:
        return build_graph().invoke({"run_id": run_id})
    except Exception as e:  # e.g. ingest service down after all retries
        log.error("run %s failed: %s", run_id, e)
        log_run({"run_id": run_id, "errors": [f"processing: {e}"],
                 "ingest": {"status": "failed", "error": str(e)}})
        return None


def main():
    global INLINE
    parser = argparse.ArgumentParser(description="News collection orchestrator")
    parser.add_argument("--schedule", action="store_true", help="keep running on a schedule")
    parser.add_argument("--inline", action="store_true", help="process in this process, no ingest service")
    parser.add_argument("--briefing", action="store_true", help="only (re)generate today's briefing")
    args = parser.parse_args()
    INLINE = args.inline

    if args.briefing:
        from rag.briefing import get_briefing
        get_briefing(force=True)
        return
    if not args.schedule:
        result = run_once()
        # non-zero exit lets GitHub Actions mark a run as failed (and e-mail you)
        raise SystemExit(0 if result and result.get("ingest", {}).get("status") in ("done", "skipped") else 1)

    from apscheduler.schedulers.blocking import BlockingScheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(run_once, "interval", hours=FETCH_EVERY_HOURS, next_run_time=datetime.now(),
                      max_instances=1, coalesce=True)
    log.info("Scheduler on: collect every %dh (briefing after %02d:00). Ctrl+C to stop.",
             FETCH_EVERY_HOURS, BRIEFING_HOUR)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
