"""
Phase 4 - Chat interface.

    venv\\Scripts\\streamlit run app/streamlit_app.py        (or double-click start_app.bat)

Tabs: Ask (chat with citations) · Today's Briefing (top stories by coverage) · Browse · Pipeline runs
Sidebar: filters (categories, optional date range), knowledge-base stats, "Collect news now", clear chat.

On Streamlit Community Cloud: main file app/streamlit_app.py; keys go in the app's Secrets
(CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE, ANTHROPIC_API_KEY or GROQ_API_KEY, GH_DISPATCH_TOKEN).
"""

import json
import os
import re
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Streamlit Cloud "Secrets" -> environment variables, before the rag modules read them
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float)):
            os.environ.setdefault(key, str(value))
except Exception:  # no secrets file locally - .env is used instead
    pass

from agents import trigger  # noqa: E402
from rag import engine, llm, store  # noqa: E402
from rag.briefing import get_briefing  # noqa: E402

st.set_page_config(page_title="News Analyst", page_icon="📰", layout="wide")

CATEGORIES = {"tech": ("💻", "Technology"), "finance": ("💹", "Finance"), "politics": ("🏛️", "Politics")}
EXAMPLES = [
    "What did the RBI announce this week and how did markets react?",
    "Any AI startup funding news in the last 3 days?",
    "What are the main political stories today?",
    "Why are bond yields rising?",
]


def label(cat):
    icon, name = CATEGORIES.get(cat, ("📰", cat))
    return f"{icon} {name}"


# ---- cached data -----------------------------------------------------------------

@st.cache_resource(show_spinner="Loading search models (first time only)...")
def warm_up():
    engine._embedder()
    engine._reranker()
    return True


@st.cache_data(ttl=300, show_spinner=False)
def kb_stats():
    return engine.stats()


@st.cache_data(ttl=300, show_spinner=False)
def all_articles():
    got = store.get_all(engine.articles_db(), ["metadatas", "documents"])
    return [{**m, "lead": d} for m, d in zip(got["metadatas"], got["documents"])]


@st.cache_data(ttl=300, show_spinner=False)
def collection_log():
    meta = engine.meta_db()
    runs, batches = store.list_records(meta, "run"), store.list_records(meta, "ingest")
    if not store.use_cloud():     # local: also show runs from before reports were kept in the knowledge base
        for rows, file, key, when in ((runs, "orchestrator_log.jsonl", "run_id", "finished"),
                                      (batches, "ingest_log.jsonl", "job_id", "time")):
            path = PROJECT_DIR / "data_chroma" / file
            if path.exists():
                known = {r.get(key) for r in rows}
                rows += [r for r in map(json.loads, filter(str.strip, path.read_text(encoding="utf-8").splitlines()))
                         if r.get(key) not in known]
                rows.sort(key=lambda r: str(r.get(when, "")), reverse=True)
        runs, batches = runs[:30], batches[:30]
    return runs, batches


@st.cache_data(ttl=60, show_spinner=False)
def github_runs():
    return trigger.recent_github_runs()


def refresh_data():
    """Pick up articles added since this app started (needed for the local store; cloud queries are live)."""
    from chromadb.api.client import SharedSystemClient
    SharedSystemClient.clear_system_cache()
    engine._collections.cache_clear()
    engine._bm25_cache["count"] = -1
    for cached in (kb_stats, all_articles, collection_log, github_runs):
        cached.clear()


# ---- small helpers --------------------------------------------------------------------

def md_safe(text):
    return text.replace("$", "\\$")          # "$30 billion" must not turn into LaTeX


def link_citations(text, sources):
    urls = {s["n"]: s["url"] for s in sources}
    return re.sub(r"\[(\d+)\]", lambda m: f"[[{m.group(1)}]]({urls[int(m.group(1))]})"
                  if int(m.group(1)) in urls else m.group(0), text)


def local_time(iso):
    """'2026-10-05T16:51:06+00:00' -> '05 Oct 22:21' in the app's time zone."""
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone().strftime("%d %b %H:%M")
    except (TypeError, ValueError):
        return str(iso)[:16]


def article_text(a, words=60):
    if a.get("summary"):
        return a["summary"]
    lead = a["lead"][len(a["title"]):].lstrip(" .") if a["lead"].startswith(a["title"]) else a["lead"]
    parts = lead.split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
        for s in sources:
            st.markdown(f"**[{s['n']}]** [{md_safe(s['title'])}]({s['url']})  \n"
                        f"{label(s['category'])} · {s['source']} · 🗓️ {s['date']}")


def show_answer(result):
    st.markdown(md_safe(link_citations(result["answer"], result["sources"])))
    show_sources(result["sources"])
    i = result["info"]
    cats = ", ".join(i["categories_filter"]) or (f"auto → {', '.join(i['categories_inferred'])} boosted"
                                                if i["categories_inferred"] else "all")
    st.caption(f"🔎 time: {i['time_label']} · categories: {cats} · {i['candidates']} chunks searched · "
               f"best match {i['best_relevance']:.2f} · {i['seconds']}s · {i['llm']}")


# ---- sidebar -------------------------------------------------------------------

warm_up()
with st.sidebar:
    st.title("📰 News Analyst")
    st.caption("Answers only from collected news · cites every claim · date-aware")

    st.subheader("Filters")
    categories = st.multiselect(
        "Categories", list(CATEGORIES), format_func=label, placeholder="All categories",
        help="Limits Ask and Browse to these domains. Leave empty for all - the domain is also guessed "
             "from your question and used as a ranking boost.")
    restrict = st.toggle("Restrict dates", help="Off: the time range comes from your question "
                                                "(\"today\", \"this week\", \"last 3 days\"). On: use a fixed range.")
    date_from = date_to = None
    now = datetime.now()
    if restrict:
        picked = st.date_input("From - to", (now.date() - timedelta(days=7), now.date()), max_value=now.date())
        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            date_from = datetime.combine(picked[0], dtime.min)
            date_to = datetime.combine(picked[1], dtime.min) + timedelta(days=1)

    st.divider()
    st.subheader("Knowledge base")
    stats = kb_stats()
    c1, c2 = st.columns(2)
    c1.metric("Articles", stats["articles"])
    c2.metric("Chunks", stats["chunks"])
    for cat in CATEGORIES:
        if cat in stats["span"]:
            lo, hi = stats["span"][cat]
            icon, name = CATEGORIES[cat]
            st.caption(f"{icon} {name.lower()}: {stats['per_category'][cat]} articles · {lo:%Y-%m-%d} → {hi:%Y-%m-%d}")
    provider, model = llm.parts()
    st.caption(f"Embeddings: `{engine.EMBED_MODEL.split('/')[-1]}`  \n"
               f"Reranker: `{engine.RERANK_MODEL.split('/')[-1]}`  \n"
               + (f"Answers: `{provider}` · `{model}`" if provider else "Answers: `no LLM key` · extractive")
               + f"  \nStore: `{store.describe()}`")

    with st.expander("Collect news now"):
        mode = trigger.mode()
        st.caption({"github": "Runs the 3 fetcher agents on GitHub Actions now (they also run 6x a day).",
                    "github-missing-token": "Runs happen 6x a day on GitHub Actions. To start one from here, "
                                            "add GH_DISPATCH_TOKEN to the app's secrets.",
                    "local": "Runs the 3 fetcher agents on this computer now."}[mode])
        if st.button("Run collection", use_container_width=True, disabled=mode == "github-missing-token"
                     or trigger.local_run_active()):
            ok, message = trigger.collect_now()
            (st.success if ok else st.error)(message)
        if mode == "local" and st.button("Reload data", use_container_width=True):
            refresh_data()
            st.rerun()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []


# ---- tabs ----------------------------------------------------------------------

tab_ask, tab_brief, tab_browse, tab_runs = st.tabs(["💬 Ask", "🌞 Today's Briefing", "📁 Browse", "⚙️ Pipeline runs"])

with tab_ask:
    st.session_state.setdefault("messages", [])
    if not st.session_state.messages:
        st.markdown("#### Ask about today's Technology, Finance and Politics news")
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES):
            if col.button(ex, use_container_width=True):
                st.session_state.pending = ex

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(md_safe(msg["content"]))
            else:
                show_answer(msg["result"])

    question = st.chat_input("e.g. What did the RBI announce this week and how did markets react?")
    question = question or st.session_state.pop("pending", None)
    if question:
        history = [(m["content"], None) for m in st.session_state.messages if m["role"] == "user"]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(md_safe(question))
        with st.chat_message("assistant"):
            with st.spinner("Searching the news..."):
                result = engine.answer(question, categories=categories or None,
                                       date_from=date_from, date_to=date_to, history=history[-2:] or None)
            show_answer(result)
        st.session_state.messages.append({"role": "assistant", "result": result})

with tab_brief:
    head, button = st.columns([4, 1], vertical_alignment="bottom")
    regenerate = button.button("Regenerate", use_container_width=True)
    with st.spinner("Building today's briefing..."):
        briefing = get_briefing(force=regenerate)
    head.header(f"Briefing for {datetime.strptime(briefing['date'], '%Y-%m-%d'):%A, %d %B %Y}")
    st.caption(f"Generated {briefing['generated_at']} · last {briefing['window_hours']} hours · "
               "ranked by how many outlets covered each story")
    sections = [(cat, briefing["sections"][cat]) for cat in CATEGORIES if cat in briefing.get("sections", {})]
    if not sections:
        st.info("No articles collected in the last 3 days yet - use **Collect news now** in the sidebar.")
    for row in range(0, len(sections), 2):
        columns = st.columns(2, gap="large")
        for col, (cat, sec) in zip(columns, sections[row:row + 2]):
            with col:
                st.subheader(label(cat))
                st.markdown(f"**{md_safe(sec['takeaway'])}**")
                for n, s in enumerate(sec["stories"], 1):
                    st.markdown(f"{n}. [{md_safe(s['title'])}]({s['url']})")
                    outlets = f" · {s['outlets']} outlets" if s["outlets"] > 1 else ""
                    st.caption(f"{s['source']} · {s['date']}{outlets}")
                    st.markdown(md_safe(s["text"]))
                    if s["related"]:
                        st.caption("Also: " + " · ".join(f"[{r['source']}]({r['url']})" for r in s["related"]))

with tab_browse:
    articles = all_articles()
    f1, f2, f3 = st.columns([3, 3, 2])
    search = f1.text_input("Search headlines and summaries", placeholder="e.g. SEBI, Nykaa, Supreme Court")
    sources = f2.multiselect("Sources", sorted({a.get("source", "") for a in articles}), placeholder="All sources")
    order = f3.selectbox("Sort", ["Newest first", "Most covered"])

    lo_ts = date_from.timestamp() if date_from else 0
    hi_ts = date_to.timestamp() if date_to else float("inf")
    terms = search.lower().split()
    shown = [a for a in articles
             if (not categories or a.get("category") in categories)
             and (not sources or a.get("source") in sources)
             and lo_ts <= a.get("published_ts", 0) < hi_ts
             and all(t in (a["title"] + " " + a.get("summary", "") + " " + a["lead"]).lower() for t in terms)]
    shown.sort(key=(lambda a: a.get("published_ts", 0)) if order == "Newest first"
               else (lambda a: (int(a.get("outlets", 1)), a.get("published_ts", 0))), reverse=True)

    limit = st.session_state.setdefault("browse_limit", 30)
    filters = [label(c) for c in categories] + (["date range"] if restrict else [])
    st.caption(f"Showing {min(limit, len(shown))} of {len(shown)} articles"
               + (f" · filtered by {', '.join(filters)}" if filters else ""))
    for a in shown[:limit]:
        when = datetime.fromtimestamp(a.get("published_ts", 0)).strftime("%Y-%m-%d %H:%M")
        outlets = f" · {a['outlets']} outlets" if int(a.get("outlets", 1)) > 1 else ""
        st.markdown(f"**[{md_safe(a['title'])}]({a['url']})**")
        st.caption(f"{label(a.get('category', ''))} · {a.get('source', '')} · {when}{outlets}")
        st.markdown(md_safe(article_text(a)))
    if len(shown) > limit and st.button("Show more"):
        st.session_state.browse_limit = limit + 30
        st.rerun()

with tab_runs:
    st.header("Pipeline runs")
    st.caption(f"Collection: fetcher agents → processing agent → {store.describe()}")
    gh = github_runs()
    if gh:
        st.markdown("**GitHub Actions** (scheduled collector)")
        st.dataframe([{**r, "started": local_time(r["started"])} for r in gh], hide_index=True,
                     use_container_width=True, column_config={"link": st.column_config.LinkColumn("link", display_text="open")})
    runs, batches = collection_log()
    st.markdown(f"**Orchestrator runs** (latest {len(runs)})")
    if runs:
        st.dataframe([{
            "finished": local_time(r.get("finished")), "where": r.get("where", "local"),
            "status": r.get("ingest", {}).get("status", "?"),
            **{f"{a} kept": v.get("kept", 0) for a, v in r.get("agents", {}).items()},
            "stored": r.get("ingest", {}).get("stored"), "pruned": r.get("housekeeping", {}).get("pruned_articles"),
            "errors": "; ".join(r.get("errors", []))[:120],
        } for r in runs], use_container_width=True, hide_index=True)
    else:
        st.caption("Nothing yet - runs appear here after the next collection.")
    st.markdown(f"**Processing batches** (latest {len(batches)}, from n8n or the orchestrator)")
    if batches:
        st.dataframe([{
            "time": local_time(r.get("time")), "received": r["received"], "stored": r["stored"],
            "duplicates": r["skipped_duplicate"], "same story elsewhere": r["skipped_near_duplicate"],
            "failed": r["failed"], "chunks": r["chunks_added"], "per category": json.dumps(r["per_category"]),
            "seconds": r["seconds"],
        } for r in batches], use_container_width=True, hide_index=True)
    else:
        st.caption("Nothing yet.")
