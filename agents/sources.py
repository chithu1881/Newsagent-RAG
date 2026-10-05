"""
Each fetcher agent's sources and relevance rules - edit this file to tune what gets collected.

  feeds   - RSS feeds (news sites + official sources such as RBI, SEBI, PIB)
  gnews   - optional GNews API topic (only used if GNEWS_API_KEY is in .env)
  strict  - True: an item must contain at least one "keep" word to be kept
  keep    - words that make an item relevant
  drop    - words that make an item irrelevant (lifestyle, sport, deals ...)

Started from the n8n "Filter & Tidy" rules (n8n/filter_tidy.js) and extended for the official feeds.
"""

AGENTS = {
    "tech": {
        "feeds": [
            ("TechCrunch", "https://techcrunch.com/feed/"),
            ("ET Tech", "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms"),
            ("The Hindu Tech", "https://www.thehindu.com/sci-tech/technology/feeder/default.rss"),
            ("Mint Technology", "https://www.livemint.com/rss/technology"),
            ("Gadgets 360", "https://feeds.feedburner.com/gadgets360-latest"),
        ],
        "gnews": "technology",
        "strict": False,
        "keep": ["ai", "artificial intelligence", "startup", "funding", "chip", "semiconductor", "software",
                 "cyber", "cloud", "data", "smartphone", "app", "telecom", "5g", "it services", "launch",
                 "openai", "google", "apple", "microsoft", "nvidia", "robot", "quantum", "regulation"],
        "drop": ["deal of the day", "discount", "sale", "giveaway", "horoscope", "price cut", "coupon",
                 "best phones under", "wordle"],
    },
    "finance": {
        "feeds": [
            ("RBI Press Releases", "https://www.rbi.org.in/pressreleases_rss.xml"),
            ("SEBI", "https://www.sebi.gov.in/sebirss.xml"),
            ("ET Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
            ("Mint Markets", "https://www.livemint.com/rss/markets"),
            ("Business Standard Markets", "https://www.business-standard.com/rss/markets-106.rss"),
            ("The Hindu Economy", "https://www.thehindu.com/business/Economy/feeder/default.rss"),
        ],
        "gnews": "business",
        "strict": True,
        "keep": ["repo rate", "rbi", "reserve bank", "sensex", "nifty", "earnings", "result", "sebi",
                 "inflation", "gdp", "ipo", "rupee", "bank", "fii", "gst", "budget", "stock", "share",
                 "market", "mutual fund", "bond", "yield", "monetary policy", "liquidity", "crude",
                 "economy", "fiscal", "tax", "investor", "circular", "regulation", "trade"],
        # last groups: routine daily RBI / SEBI notices that would crowd out real announcements
        "drop": ["lifestyle", "recipe", "horoscope", "astrology", "celebrity", "bollywood",
                 "money market operations", "variable rate repo", "variable rate reverse repo",
                 "treasury bills", "auction of state government",
                 "recovery certificate", "settlement order", "adjudication order", "order in the matter",
                 "extension of period", "section 35a"],
    },
    "politics": {
        "feeds": [
            ("PIB", "https://www.pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"),
            ("The Hindu National", "https://www.thehindu.com/news/national/feeder/default.rss"),
            ("Indian Express India", "https://indianexpress.com/section/india/feed/"),
            ("Indian Express Political Pulse", "https://indianexpress.com/section/political-pulse/feed/"),
            ("Mint Politics", "https://www.livemint.com/rss/politics"),
        ],
        "gnews": "nation",
        "strict": True,
        "keep": ["parliament", "lok sabha", "rajya sabha", "minister", "cabinet", "election", "bill",
                 "supreme court", "high court", "government", "policy", "chief minister", "opposition",
                 "party", "bjp", "congress", "governor", "assembly", "poll", "president", "ministry",
                 "scheme", "india bloc", "nda", "diplomatic", "treaty", "summit"],
        "drop": ["bollywood", "cricket", "box office", "horoscope", "recipe", "ipl", "football"],
    },
}

MAX_AGE_HOURS = 36      # ignore items older than this
SNIPPET_CHARS = 500
