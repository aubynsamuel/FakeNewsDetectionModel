import re
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from urllib.parse import urlparse
from utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS
from content_extractor import extract_content

# Configure logging
logging.basicConfig(level=logging.INFO, format=" %(message)s")

# Load SpaCy model once globally with error handling
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    logging.error(
        "⚠️ SpaCy English model 'en_core_web_md' not found. Please install: python -m spacy download en_core_web_md"
    )
    nlp = None


class ClaimVerifier:
    """Simplified and improved claim verifier for fact-checking headlines against web sources."""

    def __init__(self, cache_size: int = 500, max_workers: int = 4):
        self.claim_cache: Dict[str, Dict] = {}
        self.content_cache: Dict[str, str] = {}
        self.cache_size = cache_size
        self.max_workers = max_workers
        self.trusted_domains = TRUSTED_DOMAINS
        self.suspicious_domains = SUSPICIOUS_DOMAINS
        self.domain_weights = {"trusted": 2.0, "suspicious": 0.3, "neutral": 1.0}
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        ]
        self.current_ua_index = 1
        self.timeout = 10
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            max_df=0.9,
            max_features=5000,
            ngram_range=(1, 2),
        )

    def _get_domain_weight(self, url: str) -> Tuple[float, str]:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if domain in self.trusted_domains:
            return self.domain_weights["trusted"], "trusted"
        elif domain in self.suspicious_domains:
            return self.domain_weights["suspicious"], "suspicious"
        else:
            return self.domain_weights["neutral"], "neutral"

    def _get_user_agent(self) -> str:
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _add_to_cache(self, key: str, result: Dict):
        if len(self.claim_cache) >= self.cache_size:
            oldest_key = next(iter(self.claim_cache))
            del self.claim_cache[oldest_key]
        self.claim_cache[key] = result

    def _get_from_cache(self, key: str) -> Optional[Dict]:
        return self.claim_cache.get(key)

    def verify_claim_against_sources(
        self, claim: str, search_results: List[str]
    ) -> Dict:
        cache_key = self._cache_key(f"verify_{claim}")
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result

        support_scores = []
        contradiction_scores = []
        total_weight = 0.0
        total_sources_processed = 0
        source_details = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self._analyze_url, url, claim): url
                for url in search_results[:10]
            }
            for future in as_completed(future_to_url, timeout=45):
                url = future_to_url[future]
                try:
                    result = future.result(timeout=15)
                    if result:
                        best_score, sentiment, domain_weight, domain_type = result
                        total_sources_processed += 1
                        total_weight += domain_weight
                        # Use sentiment to adjust support/contradiction
                        if best_score >= 0.5 and sentiment >= 0:
                            support_scores.append(best_score * domain_weight)
                        elif best_score >= 0.5 and sentiment < 0:
                            contradiction_scores.append(best_score * domain_weight)
                        source_details.append(
                            {
                                "url": url,
                                "similarity": best_score,
                                "sentiment": sentiment,
                                "domain_weight": domain_weight,
                                "domain_type": domain_type,
                            }
                        )
                except Exception:
                    continue

        # Scoring logic: more support = higher score, more contradiction = lower score
        support_sum = sum(support_scores)
        contradiction_sum = sum(contradiction_scores)
        total_evidence = support_sum + contradiction_sum
        if total_sources_processed == 0 or total_evidence == 0:
            final_score = 0.1
        else:
            ratio = support_sum / total_evidence if total_evidence else 0
            if ratio >= 0.8 and support_sum >= 2.0:
                final_score = min(0.9, ratio)
            elif ratio >= 0.6:
                final_score = min(0.7, ratio)
            elif ratio >= 0.4:
                final_score = min(0.5, ratio)
            else:
                final_score = max(0.1, ratio * 0.5)

        result = {
            "score": final_score,
            "total_sources_processed": total_sources_processed,
            "support_sum": support_sum,
            "contradiction_sum": contradiction_sum,
            "total_weight": total_weight,
            "source_details": source_details,
        }
        self._add_to_cache(cache_key, result)
        return result

    def _analyze_url(
        self, url: str, claim: str
    ) -> Optional[Tuple[float, float, float, str]]:
        cache_key = self._cache_key(url)
        content = extract_content(
            url,
            self.content_cache,
            cache_key,
            self._get_user_agent,
            self.timeout,
            self.cache_size,
        )
        if not content or len(content.strip()) < 50:
            return None
        # Split content into sentences and compare claim to each
        sentences = (
            [sent.text for sent in nlp(content).sents] if nlp else content.split(".")
        )
        best_score = 0.0
        for sent in sentences:
            score = self._similarity(claim, sent)
            if score > best_score:
                best_score = score
        # Sentiment analysis (optional, fallback to 0 if not available)
        try:
            from textblob import TextBlob

            sentiment = float(TextBlob(content).sentiment.polarity)
        except Exception:
            sentiment = 0.0
        domain_weight, domain_type = self._get_domain_weight(url)
        return best_score, sentiment, domain_weight, domain_type

    def _similarity(self, text1: str, text2: str) -> float:
        if nlp and nlp(text1).has_vector and nlp(text2).has_vector:
            doc1 = nlp(text1.lower())
            doc2 = nlp(text2.lower())
            return float(doc1.similarity(doc2))
        # fallback to tfidf
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
                return 0.0
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            return 0.0
