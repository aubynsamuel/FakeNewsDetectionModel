from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from urllib.parse import urlparse
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

from deploy.utils.general_utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS
from deploy.utils.content_extractor import extract_content

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    logging.error(
        "⚠️ SpaCy English model 'en_core_web_md' not found. Please install: python -m spacy download en_core_web_md"
    )
    nlp = None


class ClaimVerifier:
    """Improved claim verifier for fact-checking headlines against web sources using paraphrase-MiniLM-L12-v2 for semantic similarity."""

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
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.196 Mobile Safari/537.36",
            "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/18.18363",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0",
        ]
        self.current_ua_index = 0
        self.timeout = 10
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            max_df=0.9,
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.sentence_transformer = None

        try:
            self.sentence_transformer = SentenceTransformer("paraphrase-MiniLM-L12-v2")
        except Exception as e:
            logging.error(
                f"❌ Failed to load paraphrase-MiniLM-L12-v2 model: {e}. Claim verification will fallback to SpaCy/TF-IDF."
            )
            self.sentence_transformer = None

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

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates semantic similarity using paraphrase-MiniLM-L12-v2 embeddings.
        Falls back to SpaCy/TF-IDF if paraphrase-MiniLM-L12-v2 is not loaded or fails.
        """
        if self.sentence_transformer:
            sentence_embeddings1 = self.sentence_transformer.encode(
                text1, show_progress_bar=False
            )
            sentence_embeddings2 = self.sentence_transformer.encode(
                text2, show_progress_bar=False
            )
            sentiment1 = TextBlob(text1).sentiment.polarity
            sentiment2 = TextBlob(text2).sentiment.polarity
            similarity = cosine_similarity(
                sentence_embeddings1.reshape(1, -1), sentence_embeddings2.reshape(1, -1)
            )[0][0]
            # Sentiment adjustment
            if sentiment1 * sentiment2 > 0:  # Both positive or both negative
                similarity = min(1.0, similarity * 1.1)
            elif sentiment1 * sentiment2 < 0:  # Opposing sentiment
                similarity = max(0.0, similarity * 0.9)
            return similarity

        if nlp and nlp(text1).has_vector and nlp(text2).has_vector:
            doc1 = nlp(text1.lower())
            doc2 = nlp(text2.lower())
            return float(doc1.similarity(doc2))

        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
                return 0.0
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            return 0.0

    def verify_claim_against_sources(
        self, claim: str, search_results: List[str]
    ) -> Dict:
        cache_key = self._cache_key(f"verify_{claim}")
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result

        support_scores = []
        total_weight = 0.0
        total_sources_processed = 0
        source_details = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self._analyze_url, url, claim): url
                for url in search_results
            }
            completed_futures = []
            try:
                for future in as_completed(future_to_url, timeout=45):
                    url = future_to_url[future]
                    try:
                        result = future.result(timeout=15)
                        if result:
                            best_similarity_score, domain_weight, domain_type = result
                            total_sources_processed += 1
                            total_weight += domain_weight

                            if best_similarity_score >= 0.5:
                                support_scores.append(
                                    best_similarity_score * domain_weight
                                )

                            source_details.append(
                                {
                                    "url": url,
                                    "semantic_similarity": best_similarity_score,
                                    "domain_weight": domain_weight,
                                    "domain_type": domain_type,
                                }
                            )
                        completed_futures.append(future)
                    except Exception as e:
                        logging.warning(f"Error processing URL {url}: {e}")
                        continue
            except TimeoutError:
                logging.warning(
                    "Timeout: Some URLs took too long to respond and were skipped."
                )

            for future in set(future_to_url) - set(completed_futures):
                url = future_to_url[future]
                logging.warning(f"Skipped slow URL (>{15}s): {url}")

        support_sum = sum(support_scores)

        if total_sources_processed == 0 or total_weight == 0:
            final_score = 0.1
        else:
            max_possible_support = total_weight
            if max_possible_support > 0:
                final_score = min(1.0, support_sum / max_possible_support)
            else:
                final_score = 0.1

            if final_score < 0.5 and support_sum < 0.5:
                final_score = max(0.1, final_score * 0.8)  # Push down more

            elif final_score > 0.5 and support_sum >= 1.0:
                final_score = min(0.9, final_score * 1.1)  # Push up more

            final_score = max(0.0, min(1.0, final_score))

        result = {
            "score": final_score,
            "total_sources_processed": total_sources_processed,
            "support_sum": support_sum,
            "total_weight": total_weight,
            "source_details": source_details,
        }
        self._add_to_cache(cache_key, result)
        # print(f"Total sources processed: {result['total_sources_processed']}")
        # print(f"Support sum: {result['support_sum']}")
        return result

    def _analyze_url(self, url: str, claim: str) -> Optional[Tuple[float, float, str]]:
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

        semantic_similarity = self._semantic_similarity(claim, content)
        # print(f"\Score from {url} : {semantic_similarity}\n")

        domain_weight, domain_type = self._get_domain_weight(url)

        return semantic_similarity, domain_weight, domain_type
