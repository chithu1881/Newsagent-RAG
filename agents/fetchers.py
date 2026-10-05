"""
Fetcher agents - one per domain (tech / finance / politics).

Each agent:
  1. reads its own feeds in parallel (3 tries per feed with back-off; one broken feed never stops the run)
  2. optionally pulls GNews headlines for its topic (if GNEWS_API_KEY is set)
  3. tidies each item  - strips HTML, cleans tracking params, converts dates to ISO
  4. applies its relevance rules - too old / drop words / off-topic / non-English / duplicates are removed
  5. returns the kept articles plus a small report of what happened

Try one agent on its own:
    venv\\Scripts\\python -m agents.fetchers finance
"""

import html
import logging
import os
import re
import sys
import time
from calendar import timegm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
from dotenv import load_dotenv

from agents.sources import AGENTS, MAX_AGE_HOURS, SNIPPET_CHARS

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsAnalystCapstone/1.0)"}
TRIES = 3
log = logging.getLogger("fetchers")


# ---- tidy helpers ----------------------------------------------------------

def clean(text):
    text = re.sub(r"<[^>]*>", " ", str(text or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def clean_url(url):
    url = str(url or "").strip().split("#")[0]
    if not re.match(r"^https?://[^\s/?#]+", url, re.I):
        return None
    if "?" not in url:
        return url
    path, query = url.split("?", 1)
    params = [p for p in query.split("&") if p and not re.match(r"^(utm_[^=]*|fbclid|gclid|ref)(=|$)", p, re.I)]
    return f"{path}?{'&'.join(params)}" if params else path


def matches(text, words):
    """Whole-word match, plural allowed: 'market' matches 'markets' but 'ai' does not match 'said'."""
    return [w for w in words if re.search(rf"\b{re.escape(w)}s?\b", text, re.I)]


def is_english(text):
    letters = [c for c in text if c.isalpha()]
    return not letters or sum(c.isascii() for c in letters) / len(letters) > 0.7


def with_retries(fn, what):
    for attempt in range(1, TRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == TRIES:
                raise
            wait = 2 ** attempt
            log.info("%s failed (%s) - retry %d in %ds", what, e, attempt, wait)
            time.sleep(wait)


# ---- the agent ---------------------------------------------------------------

class FetcherAgent:
    def __init__(self, category):
        self.category = category
        self.rules = AGENTS[category]

    def _read_feed(self, source, url):
        def get():
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.content
        parsed = feedparser.parse(with_retries(get, f"{self.category}/{source}"))
        items = []
        for e in parsed.entries:
            struct = e.get("published_parsed") or e.get("updated_parsed")
            items.append({
                "title": e.get("title", ""), "url": e.get("link") or e.get("id", ""),
                "published_ts": timegm(struct) if struct else None,
                "snippet": e.get("summary") or e.get("description", ""), "source": source,
            })
        return items

    def _read_gnews(self):
        def get():
            r = requests.get("https://gnews.io/api/v4/top-headlines", timeout=20, params={
                "category": self.rules["gnews"], "lang": "en", "country": "in", "max": 25, "apikey": GNEWS_API_KEY})
            r.raise_for_status()
            return r.json().get("articles", [])
        items = []
        for a in with_retries(get, f"{self.category}/GNews"):
            ts = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00")).timestamp() if a.get("publishedAt") else None
            items.append({"title": a.get("title", ""), "url": a.get("url", ""), "published_ts": ts,
                          "snippet": a.get("description") or a.get("content", ""),
                          "source": (a.get("source") or {}).get("name", "GNews")})
        return items

    def run(self):
        started = time.time()
        report = {"agent": self.category, "feeds_ok": 0, "feeds_failed": [], "received": 0, "broken": 0,
                  "too_old": 0, "not_english": 0, "drop_word": 0, "off_topic": 0, "duplicate": 0, "kept": 0}

        jobs = [(name, lambda n=name, u=url: self._read_feed(n, u)) for name, url in self.rules["feeds"]]
        if GNEWS_API_KEY and self.rules.get("gnews"):
            jobs.append(("GNews", self._read_gnews))

        raw = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [(name, pool.submit(fn)) for name, fn in jobs]
            for name, fut in futures:
                try:
                    raw.extend(fut.result())
                    report["feeds_ok"] += 1
                except Exception as e:
                    report["feeds_failed"].append(f"{name}: {str(e)[:80]}")
                    log.warning("[%s] feed %s failed after %d tries: %s", self.category, name, TRIES, e)

        cutoff = time.time() - MAX_AGE_HOURS * 3600
        seen_urls, seen_titles, kept = set(), set(), []
        for item in raw:
            report["received"] += 1
            title, url = clean(item["title"]), clean_url(item["url"])
            if not title or not url:
                report["broken"] += 1
                continue
            ts = item["published_ts"] or time.time()
            if ts < cutoff:
                report["too_old"] += 1
                continue
            if not is_english(title):
                report["not_english"] += 1
                continue
            snippet = clean(item["snippet"])[:SNIPPET_CHARS]
            text = f"{title} {snippet}"
            if matches(text, self.rules["drop"]):
                report["drop_word"] += 1
                continue
            hits = matches(text, self.rules["keep"])
            if self.rules["strict"] and not hits:
                report["off_topic"] += 1
                continue
            title_key = re.sub(r"[^a-z0-9 ]", "", title.lower())
            if url in seen_urls or title_key in seen_titles:
                report["duplicate"] += 1
                continue
            seen_urls.add(url)
            seen_titles.add(title_key)
            kept.append({
                "title": title, "url": url, "source": item["source"], "category": self.category,
                "published": datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z"),
                "snippet": snippet, "keywords": hits,
            })

        report["kept"] = len(kept)
        report["seconds"] = round(time.time() - started, 1)
        log.info("[%s agent] %s", self.category, report)
        return kept, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    agent = FetcherAgent(sys.argv[1] if len(sys.argv) > 1 else "finance")
    articles, rep = agent.run()
    for a in articles[:10]:
        print(f"- [{a['source']}] {a['title'][:90]}  {a['keywords'][:4]}")
    print(rep)
