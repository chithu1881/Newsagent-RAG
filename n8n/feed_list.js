// One item per RSS feed. Open each URL in your browser first; replace any that don't load.
// category must be tech / finance / politics (matches RULES in "Filter & Tidy").
const FEEDS = [
  { category: 'tech', source: 'TechCrunch', url: 'https://techcrunch.com/feed/' },
  { category: 'tech', source: 'ET Tech', url: 'https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms' },
  { category: 'finance', source: 'ET Markets', url: 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms' },
  { category: 'finance', source: 'Mint Markets', url: 'https://www.livemint.com/rss/markets' },
  { category: 'politics', source: 'The Hindu', url: 'https://www.thehindu.com/news/national/feeder/default.rss' },
  { category: 'politics', source: 'Indian Express', url: 'https://indianexpress.com/section/india/feed/' },
];

return FEEDS.map((feed) => ({ json: feed }));
