// FILTER & TIDY — turns raw RSS items into clean, tagged, relevant articles.
//
// Tidy:   strip HTML/entities, trim whitespace, clean tracking params off URLs,
//         convert dates to ISO, shorten snippets, add category + source.
// Filter: drop broken items, old items, "drop" words, off-topic items
//         (strict categories need at least one "keep" word), duplicates.

// ---- Settings you can tune ----------------------------------------------
const RULES = {
  tech: {
    strict: false, // tech feeds are already on-topic, so only the drop list applies
    keep: ['ai', 'artificial intelligence', 'startup', 'funding', 'chip', 'semiconductor', 'software',
      'cyber', 'cloud', 'data', 'smartphone', 'app', 'telecom', '5g', 'it services', 'launch'],
    drop: ['deal of the day', 'discount', 'sale', 'giveaway', 'horoscope'],
  },
  finance: {
    strict: true, // Moneycontrol "latest" mixes in everything, so require a finance word
    keep: ['repo rate', 'rbi', 'sensex', 'nifty', 'earnings', 'result', 'sebi', 'inflation', 'gdp',
      'ipo', 'rupee', 'bank', 'fii', 'gst', 'budget', 'stock', 'share', 'market', 'mutual fund'],
    drop: ['lifestyle', 'recipe', 'horoscope', 'astrology', 'celebrity'],
  },
  politics: {
    strict: true,
    keep: ['parliament', 'lok sabha', 'rajya sabha', 'minister', 'cabinet', 'election', 'bill',
      'supreme court', 'high court', 'government', 'policy', 'chief minister', 'opposition', 'party'],
    drop: ['bollywood', 'cricket', 'box office', 'horoscope', 'recipe'],
  },
};
const MAX_AGE_HOURS = 36; // ignore anything older than this
const SNIPPET_CHARS = 500;

// ---- Helpers --------------------------------------------------------------
const ENTITIES = { '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'", '&apos;': "'", '&nbsp;': ' ' };

function clean(text) {
  return String(text ?? '')
    .replace(/<!\[CDATA\[|\]\]>/g, '')
    .replace(/<[^>]*>/g, ' ')
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/&[a-z0-9#]+;/gi, (e) => ENTITIES[e.toLowerCase()] ?? ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Plain string handling on purpose: n8n Cloud's Code sandbox may not have the URL class.
function cleanUrl(raw) {
  const s = String(raw ?? '').trim();
  if (!/^https?:\/\/[^\s/?#]+/i.test(s)) return null;
  const [path, query] = s.split('#')[0].split('?');
  if (!query) return path;
  const params = query.split('&').filter((p) => p && !/^(utm_[^=]*|fbclid|gclid|ref)(=|$)/i.test(p));
  return params.length ? `${path}?${params.join('&')}` : path;
}

const hostname = (url) => url.match(/^https?:\/\/(?:www\.)?([^/?#:]+)/i)[1];

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
// Whole-word match, plural allowed: "market" matches "markets" but "ai" doesn't match "said".
const matches = (text, words) => words.filter((w) => new RegExp(`\\b${escapeRe(w)}s?\\b`, 'i').test(text));

// ---- Main -----------------------------------------------------------------
const cutoff = Date.now() - MAX_AGE_HOURS * 3600 * 1000;
const seenUrls = new Set();
const seenTitles = new Set();
const stats = { received: 0, broken: 0, tooOld: 0, dropWord: 0, offTopic: 0, duplicate: 0, kept: 0 };
const kept = [];

$input.all().forEach((item, i) => {
  stats.received++;
  const j = item.json;

  // Which feed did this item come from? (n8n links each RSS item back to its Feed List row)
  let feed = {};
  try { feed = $('Feed List').itemMatching(i).json; } catch (e) { /* fall back below */ }

  const title = clean(j.title);
  const url = cleanUrl(j.link || j.guid);
  if (!title || !url) { stats.broken++; return; } // also catches failed feeds (error items)

  const ts = Date.parse(j.isoDate || j.pubDate);
  if (!Number.isNaN(ts) && ts < cutoff) { stats.tooOld++; return; }

  const category = feed.category || 'unknown';
  const rules = RULES[category] || { strict: false, keep: [], drop: [] };
  const snippet = clean(j.contentSnippet || j.content || j.description).slice(0, SNIPPET_CHARS);
  const text = `${title} ${snippet}`;

  if (matches(text, rules.drop).length) { stats.dropWord++; return; }
  const hits = matches(text, rules.keep);
  if (rules.strict && hits.length === 0) { stats.offTopic++; return; }

  const titleKey = title.toLowerCase().replace(/[^a-z0-9 ]/g, '');
  if (seenUrls.has(url) || seenTitles.has(titleKey)) { stats.duplicate++; return; }
  seenUrls.add(url);
  seenTitles.add(titleKey);

  stats.kept++;
  kept.push({
    json: {
      title,
      url,
      published: new Date(Number.isNaN(ts) ? Date.now() : ts).toISOString(),
      source: feed.source || hostname(url),
      category,
      snippet,
      keywords: hits,
    },
  });
});

console.log('Filter & Tidy stats:', JSON.stringify(stats));

// Zero articles is never normal - fail loudly with the reason instead of silently stopping the workflow.
if (kept.length === 0) {
  const sample = $input.all()[0]?.json ?? {};
  throw new Error(`Filter & Tidy kept 0 articles. Stats: ${JSON.stringify(stats)}. `
    + `First input item fields: ${Object.keys(sample).join(', ') || 'none'}`);
}
return kept;
