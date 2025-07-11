from typing import List, Dict, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from urllib.parse import urlparse
import warnings
import re
import numpy as np
from nltk.tokenize import sent_tokenize

from deploy.utils.general_utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS
from deploy.utils.content_extractor import extract_content
from deploy.utils.entailment_analyzer_USE_lite import EntailmentAnalyzer

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(message)s")


class ClaimVerifier:
    """Enhanced claim verifier with smart sentence extraction and prioritized scraping."""

    def __init__(self, cache_size: int = 500, max_workers: int = 4):
        self.entailmentAnalyzer = EntailmentAnalyzer()
        self.use_embed = self.entailmentAnalyzer.embedder
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

    def _get_domain_weight(self, url: str) -> Tuple[float, str]:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if domain in self.trusted_domains:
            return self.domain_weights["trusted"], "trusted"
        elif domain in self.suspicious_domains:
            return self.domain_weights["suspicious"], "suspicious"
        else:
            return self.domain_weights["neutral"], "neutral"

    def _prioritize_sources(self, search_results: List[str]) -> List[str]:
        """Prioritize trusted sources if we have enough of them."""
        trusted_sources = [
            url
            for url in search_results
            if self._get_domain_weight(url)[1] == "trusted"
        ]
        other_sources = [
            url
            for url in search_results
            if self._get_domain_weight(url)[1] != "trusted"
        ]

        if len(trusted_sources) >= 4:
            return trusted_sources[:8]
        else:
            return (trusted_sources + other_sources)[:8]

    def _extract_relevant_sentences(
        self, content: str, claim: str, top_k: int = 5
    ) -> List[str]:
        """Extract the most relevant sentences from content for claim verification."""
        if not content or len(content.strip()) < 50:
            return []

        sentences = sent_tokenize(content)

        filtered_sentences = [
            s.strip()
            for s in sentences
            if 20 < len(s.strip()) < 300 and not self._is_noise_sentence(s)
        ]

        if not filtered_sentences:
            return []

        if len(filtered_sentences) <= top_k:
            return filtered_sentences

        try:
            # Use the USE Lite embedder's encode_sentence method
            claim_embedding = self.use_embed.encode_sentence(claim)

            # Use the USE Lite embedder's encode_sentences method for batch processing
            sentence_embeddings = self.use_embed.encode_sentences(filtered_sentences)

            # Calculate similarities - expand claim embedding to match sentence embeddings shape
            claim_embedding_expanded = np.expand_dims(claim_embedding, axis=0)
            similarities = np.inner(
                claim_embedding_expanded, sentence_embeddings
            ).flatten()

            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [filtered_sentences[i] for i in top_indices]
        except Exception as e:
            logging.error(f"Error in sentence ranking: {e}")
            return filtered_sentences[:top_k]

    def _is_noise_sentence(self, sentence: str) -> bool:
        """Check if a sentence is likely noise (navigation, ads, etc.)."""
        noise_patterns = [
            r"^(click|tap|read|view|see|watch|follow|subscribe)",
            r"(cookie|privacy|terms|conditions|policy)",
            r"(advertisement|sponsored|ad)",
            r"(©|copyright|\u00a9)",
            r"^(home|about|contact|menu|search)",
            r"(javascript|enable|browser|update)",
            r"^[\W\d\s]*$",
        ]
        sentence_lower = sentence.lower()
        return any(re.search(pattern, sentence_lower) for pattern in noise_patterns)

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

    def _semantic_similarity_with_sentences(
        self, claim: str, sentences: List[str]
    ) -> float:
        """Calculate entailment scores and return the best one."""
        if not sentences or not (
            self.entailmentAnalyzer.interpreter and self.entailmentAnalyzer.embedder
        ):
            return 0.1

        best_score = 0.0
        entailment_count = 0

        for sentence in sentences:
            try:
                nli_prediction = self.entailmentAnalyzer.predict_nli_tflite(
                    claim, sentence
                )
                score_map = {"Entailment": 0.95, "Neutral": 0.3, "Contradiction": 0.2}
                score = score_map.get(nli_prediction, 0.1)

                if score > best_score:
                    best_score = score
                if nli_prediction == "Entailment":
                    entailment_count += 1
                if best_score >= 0.9:
                    break
            except Exception as e:
                logging.error(f"Error analyzing sentence: {e}")
                continue

        if entailment_count > 1:
            best_score = min(0.98, best_score * 1.1)

        return best_score

    def verify_claim_against_sources(
        self, claim: str, search_results: List[str]
    ) -> Dict:
        logging.info(f"\nVerifying Claim: '{claim}'...")

        cache_key = self._cache_key(f"verify_{claim}")
        if cached_result := self._get_from_cache(cache_key):
            logging.info("📋 Using cached result")
            return cached_result

        prioritized_sources = self._prioritize_sources(search_results)

        support_scores = []
        total_weight = 0.0
        source_details = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self._analyze_url, url, claim): url
                for url in prioritized_sources
            }

            try:
                for future in as_completed(future_to_url, timeout=45):
                    url = future_to_url[future]
                    try:
                        if result := future.result(timeout=15):
                            similarity_score, domain_weight, domain_type, sentences = (
                                result
                            )

                            # Enhanced Logging Format
                            logging.info(f"\nSource: {url} ({domain_type})")
                            logging.info(
                                f"  - Relevant Sentences: {sentences[:2]}"
                            )  # Log first 2 sentences
                            logging.info(
                                f"  - Entailment Score: {similarity_score:.2f}"
                            )

                            total_weight += domain_weight
                            if similarity_score >= 0.5:
                                support_scores.append(similarity_score * domain_weight)

                            source_details.append(
                                {
                                    "url": url,
                                    "semantic_similarity": similarity_score,
                                    "domain_weight": domain_weight,
                                    "domain_type": domain_type,
                                    "relevant_sentences": sentences[:3],
                                }
                            )
                    except Exception as e:
                        logging.error(f"Error processing {url}: {e}")
            except TimeoutError:
                logging.warning("⏰ Timeout: Some URLs were skipped.")

        support_sum = sum(support_scores)

        if total_weight > 0:
            final_score = min(1.0, support_sum / total_weight)
            # Adjustments
            if final_score < 0.5 and support_sum < 0.5:
                final_score *= 0.8
            elif final_score > 0.5 and support_sum >= 1.0:
                final_score = min(0.9, final_score * 1.1)
        else:
            final_score = 0.1

        final_score = max(0.0, min(1.0, final_score))
        logging.info(
            f"\n{'='*20}\n🏁 Final Verification Score: {final_score:.2f}\n{'='*20}"
        )

        result = {
            "score": final_score,
            "total_sources_processed": len(source_details),
            "support_sum": support_sum,
            "total_weight": total_weight,
            "source_details": source_details,
        }
        self._add_to_cache(cache_key, result)
        return result

    def _analyze_url(
        self, url: str, claim: str
    ) -> Optional[Tuple[float, float, str, List[str]]]:
        try:
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

            relevant_sentences = self._extract_relevant_sentences(content, claim)

            if not relevant_sentences:
                return None

            semantic_similarity = self._semantic_similarity_with_sentences(
                claim, relevant_sentences
            )
            domain_weight, domain_type = self._get_domain_weight(url)

            return semantic_similarity, domain_weight, domain_type, relevant_sentences
        except Exception as e:
            logging.error(f"Failed to analyze URL {url}: {e}")
            return None
