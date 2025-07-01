# 🧠 SourceCredibilityAnalyzer – Domain Trustworthiness Engine

The `SourceCredibilityAnalyzer` evaluates how trustworthy a domain is by analyzing structural patterns, top-level domain (TLD) trust, and known good or bad indicators.

---

## ✅ Core Purpose

Given a domain (e.g., `cnn.com`, `abc123-fakenews.biz`), return a **normalized score between `0.0` and `1.0`**, representing how credible the domain is as a news source.

---

## 🔍 Scoring Components & Weights

| Component                    | Weight | Description                                                          |
| ---------------------------- | ------ | -------------------------------------------------------------------- |
| **TLD Credibility**          | 0.40   | Based on the top-level domain (e.g., `.gov`, `.com`, `.info`)        |
| **Domain Structure**         | 0.30   | Penalizes numeric junk, excessive hyphens, suspicious substrings     |
| **News Indicators**          | 0.20   | Bonus for known journalistic terms in domain (e.g., "news", "press") |
| **Establishment Indicators** | 0.15   | Keywords like `university`, `research`, `foundation`                 |
| **Subdomain Complexity**     | 0.10   | Penalizes overly nested subdomains (e.g., `xyz.temp.fake.news.biz`)  |

---

## 🏷 Trusted vs Suspicious Domains

- **Trusted Domains** (e.g., `bbc.com`, `nytimes.com`, `reuters.com`) → `0.95`
- **Suspicious Domains** (e.g., `infowars.com`, `naturalnews.com`) → `0.05`

These are checked first before custom scoring.

---

## 🔬 TLD-Based Scoring (`_get_tld_score()`)

Some examples:

- `.gov`, `.edu`, `.mil` → **High trust**
- `.org`, `.ac.uk`, `.edu.au` → **Moderate**
- `.info`, `.biz`, `.tk`, `.download`, `.click` → **Low or Negative Trust**

Unknown or rare TLDs default to a slight penalty (`-0.1`).

---

## 🔍 Domain Structure Patterns (`_get_structure_score()`)

Penalizes domains using:

- Long numbers (`abc1234news`)
- Suspicious words (`hoax`, `click`, `scam`, `bait`, `spam`)
- Short names (`a.co`, `xy.net`)
- Repeated symbols or excessive hyphens

Also reduces score for domains with:

- Excessive hyphens (`real-breaking-news-latest.com`)
- Gibberish-looking character patterns

---

## 📰 News Indicator Bonus (`_get_news_score()`)

Bonuses for known media terms like:

- `news`, `times`, `post`, `press`, `tribune`, `chronicle`
- Specific terms like `reuters`, `wire`, `associated`

Max cumulative bonus is **0.4**

---

## 🎓 Establishment Bonus (`_get_establishment_score()`)

Increases score for terms indicating legitimacy or research:

- `university`, `foundation`, `college`
- `research`, `science`, `archive`, `library`

Max cumulative bonus is **0.3**

---

## 🌐 Subdomain Analysis (`_get_subdomain_score()`)

- 2 parts or fewer: `+0.1` (e.g., `cnn.com`)
- 5+ parts: `-0.15` (e.g., `sub.temp.fake.news.biz`)
- Otherwise: `0.0`

---

## 📌 Example Scoring Logic

Given domain: `free-news123-clickbait.info`

1. **TLD**: `.info` → `0.1`
2. **Structure**: Contains `123`, `clickbait`, and `free` → Large penalty
3. **News Indicators**: Contains `news` → Small bonus
4. **Establishment**: None
5. **Subdomain**: Normal → Neutral
6. **Base score**: `0.2`
7. **Final Weighted Score**: ~`0.18` → Low trust

---

## 🔧 Trusted & Suspicious Lists

- `TRUSTED_DOMAINS`: Includes mainstream news (e.g., `cnn.com`, `bbc.com`, `reuters.com`), regional news (e.g., `ghanaweb.com`, `citinewsroom.com`), sports, science, and fact-checkers
- `SUSPICIOUS_DOMAINS`: Sites notorious for misinformation (e.g., `infowars.com`, `naturalnews.com`, `beforeitsnews.com`)

---

## 🧪 Final Output

```python
analyzer = SourceCredibilityAnalyzer()
score = analyzer.analyze_domain_credibility("cnn.com")
# score => 0.95
```
