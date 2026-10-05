"""
"Today's Briefing" - a short, cited morning summary per domain.

Generated once a day (07:00 by the orchestrator's scheduler, or on first open in the app) and cached in
data_chroma/briefings/YYYY-MM-DD.json, so the app never pays for it twice.

    venv\\Scripts\\python -m rag.briefing          # print today's briefing (makes it if missing)
"""

import json
import time
from datetime import datetime
from pathlib import Path

from rag import llm
from rag.engine import CATEGORIES, PROJECT_DIR, articles_db, fmt_date

BRIEFING_DIR = PROJECT_DIR / "data_chroma" / "briefings"
PER_CATEGORY = 10
WINDOWS_HOURS = (24, 72)          # try the last day; if it is thin, widen to three days

SYSTEM = """You write a crisp morning news briefing for busy Indian professionals. Today is {today}.
Use ONLY the numbered articles given. For each section (Technology, Finance, Politics) write 3-5 bullets,
most important first. Each bullet: one sentence with the key fact, then its citation like [3].
Merge articles that cover the same story into one bullet with both citations. No opinions, no outside facts.
If a section has no articles, write "_No major stories collected._" Format with markdown headings (### Technology)."""


def _recent_articles(hours):
    cutoff = int(time.time() - hours * 3600)
    db = articles_db()
    if db.count() == 0:
        return []
    got = db.get(where={"published_ts": {"$gte": cutoff}}, include=["metadatas", "documents"])
    return [{**m, "lead": d} for m, d in zip(got["metadatas"], got["documents"])]


def build_briefing():
    for hours in WINDOWS_HOURS:
        articles = _recent_articles(hours)
        if len(articles) >= 9:
            break

    picked = []
    for cat in CATEGORIES:
        in_cat = sorted((a for a in articles if a.get("category") == cat),
                        key=lambda a: a.get("published_ts", 0), reverse=True)
        picked.extend(in_cat[:PER_CATEGORY])
    sources = [{"n": i + 1, "title": a["title"], "source": a.get("source", ""), "url": a.get("url", ""),
                "category": a.get("category", ""), "date": fmt_date(a.get("published_ts", 0)),
                "text": a.get("summary") or a["lead"]} for i, a in enumerate(picked)]

    if not sources:
        markdown = "_No articles collected in the last 3 days. Run the collectors first._"
    elif llm.available():
        listing = "\n\n".join(f"[{s['n']}] ({s['category']}, {s['source']}, {s['date']}) {s['title']}\n{s['text']}"
                              for s in sources)
        markdown = llm.chat(SYSTEM.format(today=datetime.now().strftime("%A, %d %B %Y")), listing, max_tokens=1500)
    else:
        markdown = ""
    if sources and not markdown:   # no LLM (or it failed): plain headline list
        names = {"tech": "Technology", "finance": "Finance", "politics": "Politics"}
        parts = []
        for cat in CATEGORIES:
            lines = [f"- {s['title']} [{s['n']}]" for s in sources if s["category"] == cat][:5]
            parts.append(f"### {names[cat]}\n" + ("\n".join(lines) or "_No major stories collected._"))
        markdown = "\n\n".join(parts)

    return {"date": datetime.now().strftime("%Y-%m-%d"), "generated_at": datetime.now().strftime("%d %b %Y, %H:%M"),
            "window_hours": hours, "llm": llm.describe(), "markdown": markdown,
            "sources": [{k: v for k, v in s.items() if k != "text"} for s in sources]}


def get_briefing(force=False):
    path = BRIEFING_DIR / f"{datetime.now():%Y-%m-%d}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    briefing = build_briefing()
    BRIEFING_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(briefing, indent=1), encoding="utf-8")
    return briefing


if __name__ == "__main__":
    b = get_briefing()
    print(f"Briefing for {b['date']} (last {b['window_hours']}h, {b['llm']})\n")
    print(b["markdown"])
    for s in b["sources"]:
        print(f"[{s['n']}] {s['title']} - {s['source']}, {s['date']}")
