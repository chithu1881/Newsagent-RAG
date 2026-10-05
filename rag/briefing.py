"""
"Today's Briefing" - the day's top stories per domain, ranked by how many outlets covered each story.

How a story's coverage is counted:
  - at ingest, a near-identical copy from another outlet is not stored again but adds to the original's
    "outlets" count (ingest/main.py: count_extra_outlet)
  - here, articles of the last 24 h that are about the same story (embedding similarity >= SAME_STORY)
    are grouped, and their outlets are merged
The top story's headline leads each section; with an LLM key it is replaced by a one-sentence takeaway.

Built by the first collection run after BRIEFING_HOUR (or on first open in the app) and saved in the
knowledge base's "meta" collection, so every app instance - laptop or cloud - shows the same briefing.

    venv\\Scripts\\python -m rag.briefing          # print today's briefing (makes it if missing)
"""

import time
from datetime import datetime

import numpy as np

from rag import llm, store
from rag.engine import CATEGORIES, articles_db, meta_db

STORIES_PER_CATEGORY = 5
WINDOWS_HOURS = (24, 72)          # last day; widen to three days if the day is thin
SAME_STORY = 0.80                 # looser than ingest's 0.90 duplicate check: same event, different wording
SUMMARY_WORDS = 70

TAKEAWAY_SYSTEM = """You write one-sentence takeaways for a morning news briefing. Today is {today}.
For each section, write ONE plain sentence (max 30 words) summing up its most important story, using only
the headlines and summaries given. No opinions. Reply with JSON only: {{"tech": "...", "finance": "...", "politics": "..."}}
(only the sections present)."""


def _recent(hours):
    db = articles_db()
    if db.count() == 0:
        return []
    got = store.get_all(db, ["metadatas", "documents", "embeddings"],
                        where={"published_ts": {"$gte": int(time.time() - hours * 3600)}})
    return [{**m, "lead": d, "vec": np.asarray(v, dtype=float)}
            for m, d, v in zip(got["metadatas"], got["documents"], got["embeddings"])]


def _story_text(a):
    if a.get("summary"):
        return a["summary"]
    lead = a["lead"]
    if lead.startswith(a["title"]):
        lead = lead[len(a["title"]):].lstrip(" .")
    words = lead.split()
    return " ".join(words[:SUMMARY_WORDS]) + (" ..." if len(words) > SUMMARY_WORDS else "")


# Words too common in headlines to show two articles are about the same event. Measured on 5 Oct data:
# same-story pairs score 0.81-0.87 similarity, but so do unrelated "top stock picks" columns (0.79-0.82) -
# what separates them is a shared distinctive word (Nykaa, Bajaj, Gyanesh, Rajya ...).
GENERIC = set("""stock stocks share shares market markets price prices target targets nifty sensex bank banks
top buy sell picks pick outlook today tomorrow week news says said after with from what will that this
report update updates check india indian global investors investor experts expert analysts rise rises
jump jumps fall falls ahead amid over more than into could would should about their your have been
crore lakh rate rates first year years time things stop loss explained live""".split())


def _key_words(title):
    words = {w.strip("'’‘\"") for w in title.lower().replace("-", " ").split()}
    return {w for w in (x.strip(".,:;|!?()") for x in words) if len(w) >= 4 and w not in GENERIC and not w.isdigit()}


def _same_story(story, article):
    return (float(story["vec"] @ article["vec"]) >= SAME_STORY
            and bool(story["words"] & _key_words(article["title"])))


def group_stories(articles):
    """Greedy grouping, newest first: an article joins the first story it matches."""
    stories = []
    for a in sorted(articles, key=lambda x: x.get("published_ts", 0), reverse=True):
        home = next((s for s in stories if _same_story(s, a)), None)
        if home:
            home["members"].append(a)
            home["words"] |= _key_words(a["title"])
        else:
            stories.append({"vec": a["vec"], "members": [a], "words": _key_words(a["title"])})

    out = []
    for s in stories:
        members = s["members"]
        outlets = set()
        for m in members:
            outlets.update(n for n in (m.get("outlet_names") or m.get("source", "")).split(" | ") if n)
        # lead article: most widely covered, then full text over RSS snippet, then newest
        lead = max(members, key=lambda m: (int(m.get("outlets", 1)), m.get("text_type") == "full",
                                           m.get("published_ts", 0)))
        out.append({
            "title": lead["title"], "url": lead["url"], "source": lead.get("source", ""),
            "date": datetime.fromtimestamp(lead.get("published_ts", 0)).strftime("%Y-%m-%d"),
            "published_ts": lead.get("published_ts", 0), "text": _story_text(lead),
            "outlets": max(len(outlets), max(int(m.get("outlets", 1)) for m in members)),
            "outlet_names": sorted(outlets),
            "related": [{"title": m["title"], "url": m["url"], "source": m.get("source", "")}
                        for m in members if m is not lead][:3],
        })
    return sorted(out, key=lambda s: (s["outlets"], s["published_ts"]), reverse=True)


def build_briefing():
    for hours in WINDOWS_HOURS:
        articles = _recent(hours)
        if len(articles) >= 9:
            break

    sections = {}
    for cat in CATEGORIES:
        stories = group_stories([a for a in articles if a.get("category") == cat])[:STORIES_PER_CATEGORY]
        if stories:
            sections[cat] = {"takeaway": stories[0]["title"], "stories": stories}

    if sections and llm.available():
        listing = "\n\n".join(f"[{cat}]\n" + "\n".join(f"- {s['title']} ({s['outlets']} outlets): {s['text'][:300]}"
                                                       for s in sec["stories"][:3])
                              for cat, sec in sections.items())
        reply = llm.chat(TAKEAWAY_SYSTEM.format(today=datetime.now().strftime("%A, %d %B %Y")), listing, max_tokens=400)
        try:
            import json
            import re
            takeaways = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
            for cat, line in takeaways.items():
                if cat in sections and isinstance(line, str) and line.strip():
                    sections[cat]["takeaway"] = line.strip()
        except (AttributeError, ValueError):
            pass   # keep the headline takeaways

    return {"date": datetime.now().strftime("%Y-%m-%d"), "generated_at": datetime.now().strftime("%H:%M"),
            "generated_ts": int(time.time()), "window_hours": hours, "llm": llm.describe(),
            "story_count": sum(len(s["stories"]) for s in sections.values()), "sections": sections}


def get_briefing(force=False):
    record_id = f"briefing-{datetime.now():%Y-%m-%d}"
    if not force:
        cached = store.get_record(meta_db(), record_id)
        # an empty briefing (made before any news arrived) is not worth keeping for the whole day
        if cached and cached.get("story_count"):
            return cached
    briefing = build_briefing()
    store.put_record(meta_db(), record_id, "briefing", briefing)
    return briefing


def needs_morning_briefing(briefing_hour):
    """True once per day: after briefing_hour, if today's briefing is missing or was made before that hour."""
    now = datetime.now()
    if now.hour < briefing_hour:
        return False
    cached = store.get_record(meta_db(), f"briefing-{now:%Y-%m-%d}")
    morning = now.replace(hour=briefing_hour, minute=0, second=0, microsecond=0).timestamp()
    return not cached or cached.get("generated_ts", 0) < morning


if __name__ == "__main__":
    b = build_briefing()
    print(f"Briefing for {b['date']} (last {b['window_hours']}h, {b['llm']})")
    for cat, sec in b["sections"].items():
        print(f"\n## {cat}: {sec['takeaway']}")
        for i, s in enumerate(sec["stories"], 1):
            print(f"  {i}. {s['title'][:90]}  [{s['source']} · {s['date']} · {s['outlets']} outlets]")
            for r in s["related"]:
                print(f"       also: {r['source']}: {r['title'][:70]}")
