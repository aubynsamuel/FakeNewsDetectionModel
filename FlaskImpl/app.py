from flask import Flask, render_template, request, jsonify
import time
import random
from googlesearch import search
import tldextract
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np

# Download required NLTK data (run once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

app = Flask(__name__)

TRUSTED_DOMAINS = {
    # International
    "reuters.com",
    "bbc.com",
    "apnews.com",
    "nytimes.com",
    "cnn.com",
    "npr.org",
    "theguardian.com",
    "bloomberg.com",
    "dw.com",
    "washingtonpost.com",
    "wsj.com",
    "forbes.com",
    "aljazeera.com",
    "abcnews.go.com",
    "cbsnews.com",
    "usatoday.com",
    "latimes.com",
    "euronews.com",
    "indiatimes.com",
    "cbc.ca",
    "smh.com.au",
    "japantimes.co.jp",
    "economist.com",
    "ft.com",
    # Ghana-specific
    "ghanaweb.com",
    "myjoyonline.com",
    "graphic.com.gh",
    "citinewsroom.com",
    "3news.com",
    "peacefmonline.com",
    "pulse.com.gh",
    "adomonline.com",
    "starrfm.com.gh",
    "modernghana.com",
    "ghanaiantimes.com.gh",
    "yen.com.gh",
    "ghananewsagency.org",
    "thebftonline.com",
    # Sports
    "espn.com",
    "skysports.com",
    "goal.com",
    "sportinglife.com",
    "supersport.com",
    "onefootball.com",
    "cbssports.com",
    "nbcsports.com",
    "foxsports.com",
    "eurosport.com",
    "theathletic.com",
    "ghanasoccernet.com",
    "footballghana.com",
    # Entertainment
    "variety.com",
    "hollywoodreporter.com",
    "entertainment.com",
    "eonline.com",
    "people.com",
    "ew.com",
    "deadline.com",
    "indiewire.com",
    "rollingstone.com",
    "thewrap.com",
    "billboard.com",
}

# Suspicious keywords that often appear in fake news
FAKE_NEWS_INDICATORS = [
    "shocking",
    "unbelievable",
    "scientists hate this",
    "doctors hate this",
    "this one trick",
    "you won't believe",
    "breaking: urgent",
    "leaked footage",
    "hidden truth",
    "they don't want you to know",
    "miracle cure",
    "instant results",
    "amazing discovery",
]


class NewsContentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=100)

    def extract_article_text(self, url, timeout=8):
        """Extract main text content from a news article"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Try to find article content using common tags
            content = ""
            for tag in [
                "article",
                "main",
                'div[class*="content"]',
                'div[class*="article"]',
            ]:
                elements = soup.select(tag)
                if elements:
                    content = " ".join([elem.get_text() for elem in elements])
                    break

            if not content:
                content = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = " ".join(chunk for chunk in chunks if chunk)

            return content[:1500]  # Limit to first 1500 characters for speed
        except Exception as e:
            return ""

    def calculate_content_similarity(self, headline, content):
        """Calculate similarity between headline and article content"""
        if not content or len(content) < 50:
            return 0.0

        try:
            documents = [headline.lower(), content.lower()]
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0

    def analyze_sentiment_consistency(self, headline, content):
        """Check if headline and content have consistent sentiment"""
        if not content:
            return 0.5

        headline_sentiment = TextBlob(headline).sentiment.polarity
        content_sentiment = TextBlob(content[:500]).sentiment.polarity

        sentiment_diff = abs(headline_sentiment - content_sentiment)
        consistency = max(0, 1 - sentiment_diff)

        return consistency

    def check_suspicious_language(self, headline):
        """Check for suspicious language patterns in headline"""
        headline_lower = headline.lower()
        suspicious_count = sum(
            1 for indicator in FAKE_NEWS_INDICATORS if indicator in headline_lower
        )

        # Check for excessive punctuation or caps
        exclamation_count = headline.count("!")
        caps_ratio = (
            sum(1 for c in headline if c.isupper()) / len(headline) if headline else 0
        )

        suspicious_score = (
            suspicious_count * 0.3 + min(exclamation_count, 3) * 0.1 + caps_ratio * 0.2
        )

        return min(suspicious_score, 1.0)


def extract_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"


def analyze_news_headline(headline, results_to_check=6, min_trusted=2):
    """Main function to analyze news headline - returns structured data"""

    analyzer = NewsContentAnalyzer()

    # Check for suspicious language patterns
    suspicious_score = analyzer.check_suspicious_language(headline)

    # Add delay to avoid being blocked
    time.sleep(random.uniform(1, 2))

    try:
        # Perform Google search
        search_results = list(search(headline, num_results=results_to_check, lang="en"))
    except Exception as e:
        return {"error": f"Search failed: {str(e)}", "headline": headline}

    # Analyze search results
    trusted_hits = []
    content_similarities = []
    sentiment_consistencies = []
    source_details = []

    for i, url in enumerate(search_results[:results_to_check]):
        domain = extract_domain(url)
        is_trusted = domain in TRUSTED_DOMAINS

        source_info = {
            "domain": domain,
            "url": url,
            "is_trusted": is_trusted,
            "similarity": 0,
            "sentiment_consistency": 0.5,
        }

        if is_trusted:
            trusted_hits.append(domain)

            # Extract and analyze content (limit to avoid long processing)
            content = analyzer.extract_article_text(url)
            if content:
                similarity = analyzer.calculate_content_similarity(headline, content)
                sentiment_consistency = analyzer.analyze_sentiment_consistency(
                    headline, content
                )

                content_similarities.append(similarity)
                sentiment_consistencies.append(sentiment_consistency)

                source_info["similarity"] = similarity
                source_info["sentiment_consistency"] = sentiment_consistency

        source_details.append(source_info)

    # Calculate final scores
    unique_trusted = set(trusted_hits)
    avg_similarity = np.mean(content_similarities) if content_similarities else 0
    avg_sentiment = np.mean(sentiment_consistencies) if sentiment_consistencies else 0.5

    # Enhanced scoring system
    trust_score = min(len(unique_trusted) / min_trusted, 1.0)
    content_score = (avg_similarity + avg_sentiment) / 2
    language_score = 1 - suspicious_score

    # Weighted final score
    final_score = trust_score * 0.4 + content_score * 0.4 + language_score * 0.2

    # Determine verdict
    if final_score >= 0.7:
        verdict = "✅ LIKELY REAL"
        verdict_class = "real"
        confidence = "High"
    elif final_score >= 0.5:
        verdict = "⚠️ POSSIBLY REAL"
        verdict_class = "maybe"
        confidence = "Medium"
    elif final_score >= 0.3:
        verdict = "❓ UNCERTAIN"
        verdict_class = "uncertain"
        confidence = "Low"
    else:
        verdict = "🚫 LIKELY FAKE"
        verdict_class = "fake"
        confidence = "High"

    return {
        "headline": headline,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "confidence": confidence,
        "final_score": round(final_score, 2),
        "metrics": {
            "trusted_sources": len(unique_trusted),
            "content_similarity": round(avg_similarity, 2),
            "sentiment_consistency": round(avg_sentiment, 2),
            "suspicious_score": round(suspicious_score, 2),
        },
        "sources": source_details[:8],  # Limit sources shown
        "total_results": len(search_results),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        headline = data.get("headline", "").strip()

        if not headline:
            return jsonify({"error": "Please enter a headline to analyze"})

        if len(headline) < 10:
            return jsonify(
                {"error": "Headline too short. Please enter a more complete headline."}
            )

        # Analyze the headline
        result = analyze_news_headline(headline)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
