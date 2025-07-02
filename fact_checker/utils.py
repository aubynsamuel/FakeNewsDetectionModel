import tldextract
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Enhanced trusted domains (your original + more)
TRUSTED_DOMAINS = {
    # 🌍 International Mainstream News
    "abcnews.go.com",
    "aljazeera.com",
    "apnews.com",
    "bbc.com",
    "bloomberg.com",
    "cbc.ca",
    "cbsnews.com",
    "cnn.com",
    "dw.com",
    "economist.com",
    "euronews.com",
    "forbes.com",
    "ft.com",
    "indiatimes.com",
    "japantimes.co.jp",
    "latimes.com",
    "npr.org",
    "nytimes.com",
    "reuters.com",
    "smh.com.au",
    "theguardian.com",
    "usatoday.com",
    "washingtonpost.com",
    "wsj.com",
    "france24.com",
    # 📰 Ghana-Specific News
    "3news.com",
    "adomonline.com",
    "citinewsroom.com",
    "ghanaweb.com",
    "ghanaiantimes.com.gh",
    "ghananewsagency.org",
    "graphic.com.gh",
    "modernghana.com",
    "myjoyonline.com",
    "peacefmonline.com",
    "pulse.com.gh",
    "starrfm.com.gh",
    "thebftonline.com",
    "yen.com.gh",
    # ⚽ Sports News
    "cbssports.com",
    "espn.com",
    "eurosport.com",
    "fifa.com",
    "footballghana.com",
    "foxsports.com",
    "ghanasoccernet.com",
    "goal.com",
    "nba.com",
    "nbcsports.com",
    "onefootball.com",
    "skysports.com",
    "sportinglife.com",
    "supersport.com",
    "tntsports.co.uk",
    "theathletic.com",
    "olympics.com",
    # 🎬 Entertainment & Pop Culture
    "billboard.com",
    "deadline.com",
    "entertainment.com",
    "eonline.com",
    "ew.com",
    "hollywoodreporter.com",
    "indiewire.com",
    "people.com",
    "rollingstone.com",
    "thewrap.com",
    "variety.com",
    # 🧪 Science & Research
    "eurekalert.org",
    "medpagetoday.com",
    "nasa.gov",
    "nature.com",
    "sciencealert.com",
    "sciencenews.org",
    "statnews.com",
    # 🌐 Fact-Checking & Watchdogs
    "africacheck.org",
    "factcheck.org",
    "fullfact.org",
    "politifact.com",
    "snopes.com",
    # 🌍 Global & General Niche News
    "asia.nikkei.com",
    "globalissues.org",
    "ipsnews.net",
    "oecdobserver.org",
    "rferl.org",
    # 📰 African Regional News (non-Ghana)
    "dailynation.africa",
    "enca.com",
    "ewn.co.za",
    "monitor.co.ug",
    "thecitizen.co.tz",
    "businessinsider.com",
    "africanews.com",
    # 🎓 Academic & Policy Think Tanks
    "brookings.edu",
    "carnegieendowment.org",
    "cfr.org",
    "foreignpolicy.com",
    "theconversation.com",
}

# Suspicious domains that often spread misinformation
SUSPICIOUS_DOMAINS = {
    "beforeitsnews.com",
    "naturalnews.com",
    "infowars.com",
    "breitbart.com",
    "dailystormer.com",
    "zerohedge.com",
    "activistpost.com",
    "realfarmacy.com",
    "healthnutnews.com",
}


def extract_domain(url):
    """Extract domain from URL"""
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"


_PATTERNS = [
    (re.compile(r"\b[A-Z]+\s*\(Reuters\)\s*[-–—]?\s*", re.IGNORECASE), ""),
    (re.compile(r"\(Reuters\)", re.IGNORECASE), ""),
    (re.compile(r"Reuters", re.IGNORECASE), ""),
    (
        re.compile(
            r"\b(?:WASHINGTON|NEW YORK|LONDON|PARIS|BERLIN|TOKYO|MOSCOW|BEIJING|DELHI)\s*[-–—]?\s*",
            re.IGNORECASE,
        ),
        "",
    ),
    (re.compile(r"\b(?:AP|CNN|BBC|Fox News|NBC|CBS|ABC News)\b", re.IGNORECASE), ""),
    (re.compile(r"\bBy\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", re.IGNORECASE), ""),
    (re.compile(r"\S+@\S+\.\S+"), ""),
    (re.compile(r"http[s]?://\S+"), ""),
    (re.compile(r"[^a-zA-Z\s]"), " "),
    (re.compile(r"\s+"), " "),
]


def remove_source_artifacts_fast(text):
    """Optimized version of source artifact removal"""
    if not isinstance(text, str) or len(text) < 10:
        return ""

    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)

    return text.strip().lower()


def _process_text_chunk(text_chunk):
    """Internal helper to process a chunk of texts in parallel"""
    return [remove_source_artifacts_fast(text) for text in text_chunk]


def parallel_preprocess(texts, n_jobs=None):
    """Parallel preprocessing of texts using multiprocessing"""
    if n_jobs is None:
        n_jobs = min(cpu_count(), 8)

    chunk_size = max(1, len(texts) // n_jobs)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    print(
        f"Processing {len(texts)} texts in {len(chunks)} chunks using {n_jobs} processes..."
    )

    with Pool(n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(_process_text_chunk, chunks),
                total=len(chunks),
                desc="Preprocessing chunks",
            )
        )

    processed_texts = []
    for chunk_result in results:
        processed_texts.extend(chunk_result)

    return processed_texts
