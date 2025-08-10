from typing import List, Dict, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from urllib.parse import urlparse
import warnings
import re
from nltk.tokenize import sent_tokenize
import string

from deploy.utils.general_utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS
from deploy.utils.content_extractor import extract_content
from deploy.utils.url_filter import _is_corrupted_pdf_content, _is_pdf_or_download_url
from semantic_similarity import calculate_semantic_similarity

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(message)s")


class ClaimVerifier:
    """Enhanced claim verifier with smart sentence extraction and prioritized scraping."""

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

    def _get_domain_weight(self, url: str) -> Tuple[float, str]:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if domain in self.trusted_domains:
            return self.domain_weights["trusted"], "trusted"
        elif domain in self.suspicious_domains:
            return self.domain_weights["suspicious"], "suspicious"
        else:
            return self.domain_weights["neutral"], "neutral"

    def _prioritize_sources(self, search_results: List[str]) -> List[str]:
        """Prioritize trusted sources and filter out PDFs/downloads."""
        # First, filter out PDFs and download links
        filtered_results = []
        pdf_count = 0

        for url in search_results:
            if _is_pdf_or_download_url(url):
                pdf_count += 1
                logging.info(f"📄 Filtered out PDF/download URL: {url}")
                continue
            filtered_results.append(url)

        if pdf_count > 0:
            logging.info(f"🚫 Filtered out {pdf_count} PDF/download URLs")

        if not filtered_results:
            logging.warning("⚠️ No valid URLs remaining after filtering PDFs/downloads")
            return []

        # Then prioritize trusted sources
        trusted_sources = [
            url
            for url in filtered_results
            if self._get_domain_weight(url)[1] == "trusted"
        ]
        other_sources = [
            url
            for url in filtered_results
            if self._get_domain_weight(url)[1] != "trusted"
        ]

        if len(trusted_sources) >= 4:
            return trusted_sources[:8]
        else:
            return (trusted_sources + other_sources)[:8]

    def _is_valid_sentence(self, sentence: str) -> bool:
        """Enhanced sentence validation to filter out garbled/corrupted text."""
        sentence = sentence.strip()

        # Basic length check
        if len(sentence) < 20 or len(sentence) > 300:
            return False

        # Check for too many non-ASCII characters (garbled text indicator)
        non_ascii_count = sum(1 for c in sentence if ord(c) > 127)
        if non_ascii_count > len(sentence) * 0.3:  # More than 30% non-ASCII
            return False

        # Check for excessive special characters or symbols
        special_chars = sum(
            1 for c in sentence if c in string.punctuation and c not in ".,!?;:"
        )
        if special_chars > len(sentence) * 0.2:  # More than 20% special chars
            return False

        # Enhanced check for random character patterns (PDF corruption indicators)
        if re.search(r"[^\w\s]{3,}", sentence):  # 3+ consecutive non-word chars
            return False

        # Check for PDF-specific corruption patterns
        if re.search(r"(endstream|endobj|obj\s*<|stream\s+H)", sentence, re.IGNORECASE):
            return False

        # Check for excessive whitespace or control characters
        if re.search(r"\s{3,}", sentence) or any(
            ord(c) < 32 and c not in "\t\n\r" for c in sentence
        ):
            return False

        # Check for minimum word count and average word length
        words = sentence.split()
        if len(words) < 4:
            return False

        # Check for reasonable word lengths (avoid strings like "a b c d e f g")
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 2.5:
            return False

        # Check for excessive capitalization
        if sum(1 for c in sentence if c.isupper()) > len(sentence) * 0.5:
            return False

        # Check for sequences that look like corrupted encoding
        if re.search(r"[^\w\s]{5,}", sentence):
            return False

        return True

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
            r"(share|like|comment|subscribe)",
            r"(login|sign\s+in|register)",
            r"(loading|please\s+wait)",
            # Add PDF-specific noise patterns
            r"(pdf|download|file|document)\s*(viewer|reader)",
            r"(page|pages)\s*\d+\s*(of|\/)\s*\d+",
            r"(adobe|acrobat|reader)",
        ]
        sentence_lower = sentence.lower()
        return any(re.search(pattern, sentence_lower) for pattern in noise_patterns)

    def _extract_relevant_sentences(self, content: str) -> List[str]:
        """Extract relevant sentences using TF-IDF vectorization."""
        if not content or len(content.strip()) < 50:
            return []

        # Check if content appears to be corrupted PDF
        if _is_corrupted_pdf_content(content):
            logging.warning("🚫 Content appears to be corrupted PDF - skipping")
            return []

        sentences = sent_tokenize(content)

        # Enhanced filtering pipeline
        valid_sentences = []
        for sentence in sentences:
            if self._is_valid_sentence(sentence) and not self._is_noise_sentence(
                sentence
            ):
                valid_sentences.append(sentence.strip())

        if not valid_sentences:
            logging.warning("No valid sentences found after filtering")
            return []

        return valid_sentences

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

    def _semantic_similarity_with_sentences(self, claim: str, sentences: str) -> float:
        """Calculate entailment scores and return the best one."""
        try:
            score = calculate_semantic_similarity(claim, sentences)
        except Exception as e:
            logging.error(f"Error analyzing sentence: {e}")
        return score

    def verify_claim_against_sources(
        self, claim: str, search_results: List[str]
    ) -> Dict:
        logging.info(f"\nVerifying Claim: '{claim}'...")

        cache_key = self._cache_key(f"verify_{claim}")
        if cached_result := self._get_from_cache(cache_key):
            logging.info("📋 Using cached result")
            return cached_result

        prioritized_sources = self._prioritize_sources(search_results)

        if not prioritized_sources:
            logging.warning("⚠️ No valid sources available after filtering")
            return {
                "score": 0.3,
                "total_sources_processed": 0,
                "support_sum": 0.0,
                "total_weight": 0.0,
                "source_details": [],
                "warning": "No valid sources available after filtering PDFs/downloads",
            }

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

                            total_weight += domain_weight
                            if similarity_score >= 0.4:
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

        # for source_detail in source_details:
        #     logging.info(f"Source Details:\n{source_detail}\n")

        support_sum = sum(support_scores)

        if total_weight > 0:
            final_score = min(1.0, support_sum / total_weight)
            # Adjustments
            if final_score < 0.5:
                final_score *= 0.9
            elif final_score > 0.5:
                final_score *= 1.1
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
            # Double-check for PDFs at analysis time (in case some slipped through)
            if _is_pdf_or_download_url(url):
                logging.info(f"🚫 Skipping PDF/download URL at analysis time: {url}")
                return None

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

            # Check for corrupted PDF content
            if _is_corrupted_pdf_content(content):
                logging.warning(f"🚫 Skipping corrupted PDF content from: {url}")
                return None

            # Used for sentence extraction instead of embeddings
            relevant_sentences = self._extract_relevant_sentences(content)

            if not relevant_sentences:
                return None

            cleaned_content = ""
            for sentence in relevant_sentences:
                if (
                    sentence.endswith(".")
                    or sentence.endswith("?")
                    or sentence.endswith("!")
                ):
                    cleaned_content += f"{sentence} "
                else:
                    cleaned_content += f"{sentence}. "

            semantic_similarity = self._semantic_similarity_with_sentences(
                claim, cleaned_content
            )

            domain_weight, domain_type = self._get_domain_weight(url)
            # print(f"relevant_sentences: {cleaned_content}")

            return semantic_similarity, domain_weight, domain_type, relevant_sentences
        except Exception as e:
            logging.error(f"Failed to analyze URL {url}: {e}")
            return None
