"""
Phase 4 - Chat interface.

    venv\\Scripts\\streamlit run app/streamlit_app.py        (or double-click start_app.bat)

Tabs: Ask (chat with citations) · Today's Briefing · Collection log
Sidebar: knowledge-base stats, category filter, date filter.
"""

import json
import re
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from rag import engine, llm  # noqa: E402
from rag.briefing import get_briefing  # noqa: E402

st.set_page_config(page_title="News Analyst", page_icon="📰", layout="wide")

CATEGORY_LABELS = {"tech": "💻 Technology", "finance": "💹 Finance", "politics": "🏛️ Politics"}
EXAMPLES = [
    "What did the RBI announce this week and how did markets react?",
    "Any AI startup funding news in the last 3 days?",
    "What are the main political stories today?",
    "Why are bond yields rising?",
]


@st.cache_resource(show_spinner="Loading search models (first time only)...")
def warm_up():
    engine._embedder()
    engine._reranker()
    return True


def refresh_data():
    """Pick up articles the ingest service added since this app started."""
    from chromadb.api.client import SharedSystemClient
    SharedSystemClient.clear_system_cache()
    engine._collections.cache_clear()
    engine._bm25_cache["count"] = -1


def md_safe(text):
    return text.replace("$", "\\$")          # "$30 billion" must not turn into LaTeX


def link_citations(text, sources):
    urls = {s["n"]: s["url"] for s in sources}
    return re.sub(r"\[(\d+)\]", lambda m: f"[[{m.group(1)}]]({urls[int(m.group(1))]})"
                  if int(m.group(1)) in urls else m.group(0), text)


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
        for s in sources:
            st.markdown(f"**[{s['n']}]** [{md_safe(s['title'])}]({s['url']})  \n"
                        f"{CATEGORY_LABELS.get(s['category'], s['category'])} · {s['source']} · 🗓️ {s['date']}")


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
    st.caption("Answers only from collected news, with sources and dates.")

    if st.button("🔄 Refresh data", use_container_width=True):
        refresh_data()
    stats = engine.stats()
    c1, c2 = st.columns(2)
    c1.metric("Articles", stats["articles"])
    c2.metric("Chunks", stats["chunks"])
    st.caption(" · ".join(f"{CATEGORY_LABELS[c]} {stats['per_category'].get(c, 0)}" for c in CATEGORY_LABELS))
    if stats["newest"]:
        st.caption(f"Newest article: {stats['newest']:%d %b %Y, %H:%M}")
    st.caption(f"LLM: {llm.describe()}")

    st.divider()
    st.subheader("Filters")
    categories = st.multiselect("Categories", list(CATEGORY_LABELS), format_func=CATEGORY_LABELS.get,
                                placeholder="All (auto-detected from question)")
    date_mode = st.radio("Dates", ["Auto (from question)", "Last 24 hours", "Last 7 days", "Custom range"])
    date_from = date_to = None
    now = datetime.now()
    if date_mode == "Last 24 hours":
        date_from = now - timedelta(hours=24)
    elif date_mode == "Last 7 days":
        date_from = now - timedelta(days=7)
    elif date_mode == "Custom range":
        picked = st.date_input("From - to", (now.date() - timedelta(days=7), now.date()), max_value=now.date())
        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            date_from = datetime.combine(picked[0], dtime.min)
            date_to = datetime.combine(picked[1], dtime.min) + timedelta(days=1)

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []


# ---- tabs ----------------------------------------------------------------------

tab_ask, tab_brief, tab_log = st.tabs(["💬 Ask", "☀️ Today's Briefing", "📊 Collection log"])

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
    top = st.columns([4, 1])
    regenerate = top[1].button("↻ Regenerate", use_container_width=True)
    with st.spinner("Writing today's briefing..."):
        briefing = get_briefing(force=regenerate)
    top[0].markdown(f"### ☀️ Briefing for {datetime.strptime(briefing['date'], '%Y-%m-%d'):%A, %d %B %Y}")
    st.caption(f"Generated {briefing['generated_at']} from news of the last {briefing['window_hours']} hours · "
               f"{briefing['llm']}")
    st.markdown(md_safe(link_citations(briefing["markdown"], briefing["sources"])))
    show_sources(briefing["sources"])

with tab_log:
    st.markdown("### What was collected")
    for name, label in (("orchestrator_log.jsonl", "Python orchestrator runs"),
                        ("ingest_log.jsonl", "Ingest batches (n8n or orchestrator)")):
        path = PROJECT_DIR / "data_chroma" / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
            if path.exists() else []
        st.markdown(f"**{label}** ({len(rows)})")
        if not rows:
            st.caption("Nothing yet.")
        elif name.startswith("orchestrator"):
            st.dataframe([{
                "run": r["run_id"], "status": r.get("ingest", {}).get("status", "?"),
                **{f"{a} kept": v.get("kept", 0) for a, v in r.get("agents", {}).items()},
                "stored": r.get("ingest", {}).get("stored"), "errors": "; ".join(r.get("errors", []))[:120],
            } for r in reversed(rows[-30:])], use_container_width=True, hide_index=True)
        else:
            st.dataframe([{
                "time": r["time"][:16].replace("T", " "), "received": r["received"], "stored": r["stored"],
                "duplicates": r["skipped_duplicate"], "near-dupes": r["skipped_near_duplicate"],
                "failed": r["failed"], "chunks": r["chunks_added"], "per category": json.dumps(r["per_category"]),
                "seconds": r["seconds"],
            } for r in reversed(rows[-30:])], use_container_width=True, hide_index=True)
