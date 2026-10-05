"""
"Collect news now" from the app.

  Cloud app (news from GitHub) -> starts the GitHub Actions workflow (collect.yml) through the GitHub API.
                                Needs GH_DISPATCH_TOKEN: a fine-grained token with "Actions: read and write"
                                on the repo; GH_REPO defaults to chithu1881/Newsagent-RAG.
  Local store                -> runs the orchestrator as a background process on this computer
                                (through the ingest service if it is running, otherwise --inline).
"""

import os
import subprocess
import sys
from pathlib import Path

import requests

from rag import store

PROJECT_DIR = Path(__file__).resolve().parent.parent
GH_API = "https://api.github.com"
WORKFLOW = "collect.yml"
_local_run = {"proc": None}


def _gh():
    token = os.getenv("GH_DISPATCH_TOKEN", "")
    repo = os.getenv("GH_REPO", "chithu1881/Newsagent-RAG")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    return token, repo, headers


def mode():
    """'github', 'github-missing-token' or 'local'."""
    if store.source() == "github":
        return "github" if os.getenv("GH_DISPATCH_TOKEN") else "github-missing-token"
    return "local"


def collect_now():
    """Start a collection run. Returns (ok, message)."""
    if mode() == "github-missing-token":
        return False, "Add GH_DISPATCH_TOKEN to the app's secrets to start runs from here."
    if mode() == "github":
        token, repo, headers = _gh()
        r = requests.post(f"{GH_API}/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches",
                          headers=headers, json={"ref": "main"}, timeout=20)
        if r.status_code == 204:
            return True, "Started on GitHub Actions. New articles appear here in about 20 minutes."
        return False, f"GitHub refused the request ({r.status_code}): {r.text[:150]}"

    proc = _local_run["proc"]
    if proc is not None and proc.poll() is None:
        return False, "A collection run is already in progress."
    try:
        service_up = requests.get(os.getenv("INGEST_URL", "http://127.0.0.1:8000") + "/health", timeout=2).ok
    except requests.RequestException:
        service_up = False
    args = [sys.executable, "-m", "agents.orchestrator"] + ([] if service_up else ["--inline"])
    log = open(PROJECT_DIR / "data_chroma" / "collect_now.log", "w", encoding="utf-8")
    _local_run["proc"] = subprocess.Popen(args, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)
    return True, "Collecting on this computer. It takes about 5 minutes; then press Reload data."


def local_run_active():
    proc = _local_run["proc"]
    return proc is not None and proc.poll() is None


def recent_github_runs(n=5):
    """Latest workflow runs, or [] when there is no token / the API is unreachable."""
    token, repo, headers = _gh()
    if not token:
        return []
    try:
        r = requests.get(f"{GH_API}/repos/{repo}/actions/workflows/{WORKFLOW}/runs",
                         headers=headers, params={"per_page": n}, timeout=15)
        r.raise_for_status()
    except requests.RequestException:
        return []
    return [{"started": run["run_started_at"], "trigger": run["event"], "status": run["status"],
             "result": run["conclusion"] or "-", "link": run["html_url"]} for run in r.json().get("workflow_runs", [])]
