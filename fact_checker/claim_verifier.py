from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from urllib.parse import urlparse
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import warnings

# Assuming utils and content_extractor are available
from utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS
from content_extractor import extract_content

warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Load SpaCy model once globally (consider 'en_core_web_md' if not doing complex NER/parsing)
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    logging.error(
        "⚠️ SpaCy English model 'en_core_web_md' not found. Please install: python -m spacy download en_core_web_md"
    )
    nlp = None


class ClaimVerifier:
    """Improved claim verifier for fact-checking headlines against web sources using DistilBERT for semantic similarity."""

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
        # TF-IDF is still kept as a very basic fallback, though less important with DistilBERT embeddings
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            max_df=0.9,
            max_features=5000,
            ngram_range=(1, 2),
        )

        # DistilBERT Model Setup for Semantic Similarity
        self.distilbert_tokenizer = None
        self.distilbert_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.distilbert_model = DistilBertModel.from_pretrained(
                "distilbert-base-uncased"
            ).to(self.device)
            self.distilbert_model.eval()  # Set model to evaluation mode
            logging.info(
                f"🚀 DistilBERT model 'distilbert-base-uncased' loaded successfully on {self.device}."
            )
        except Exception as e:
            logging.error(
                f"❌ Failed to load DistilBERT model: {e}. Claim verification will fallback to SpaCy/TF-IDF."
            )
            self.distilbert_tokenizer = None
            self.distilbert_model = None

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

    def _get_sentence_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Generates a sentence embedding using DistilBERT.
        Uses the [CLS] token embedding as the sentence representation.
        """
        if not self.distilbert_model:
            return None

        inputs = self.distilbert_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)

        # Use the [CLS] token embedding as the sentence embedding
        # The output is (batch_size, sequence_length, hidden_size)
        # We want the [CLS] token which is the first token (index 0)
        # and then squeeze to remove the batch dimension if batch_size is 1
        return outputs.last_hidden_state[:, 0, :].squeeze(0)

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates semantic similarity using DistilBERT embeddings.
        Falls back to SpaCy/TF-IDF if DistilBERT is not loaded or fails.
        """
        if self.distilbert_model:
            emb1 = self._get_sentence_embedding(text1)
            emb2 = self._get_sentence_embedding(text2)

            if emb1 is not None and emb2 is not None:
                # Cosine similarity between embeddings
                return float(
                    torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
                )

        # Fallback to SpaCy/TF-IDF if DistilBERT not available or embeddings couldn't be generated
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
        # Without an NLI model, direct "contradiction" detection is hard.
        # We'll focus on the strength of "support" based on semantic similarity.
        # If a claim is contradicted, it will likely have low semantic similarity to most sources,
        # or require more advanced text analysis beyond simple similarity.
        total_weight = 0.0
        total_sources_processed = 0
        source_details = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self._analyze_url, url, claim): url
                for url in search_results[:10]  # Limit to top 10 for performance
            }
            for future in as_completed(future_to_url, timeout=45):
                url = future_to_url[future]
                try:
                    result = future.result(timeout=15)
                    if result:
                        best_similarity_score, domain_weight, domain_type = result
                        total_sources_processed += 1
                        total_weight += domain_weight

                        # Consider a semantic similarity score above a certain threshold as support
                        # This threshold will need fine-tuning based on your data
                        if (
                            best_similarity_score >= 0.7
                        ):  # A higher threshold for strong semantic support
                            support_scores.append(best_similarity_score * domain_weight)

                        source_details.append(
                            {
                                "url": url,
                                "semantic_similarity": best_similarity_score,
                                "domain_weight": domain_weight,
                                "domain_type": domain_type,
                            }
                        )
                except Exception as e:
                    logging.warning(f"Error processing URL {url}: {e}")
                    continue

        # Scoring logic: Primarily based on accumulated semantic support
        support_sum = sum(support_scores)

        # Adjust the scoring logic to reflect the confidence in the claim
        # A simple approach: Normalize the sum of weighted support scores
        if total_sources_processed == 0 or total_weight == 0:
            final_score = 0.1  # Neutral if no sources or no weight
        else:
            # Scale the support sum by the maximum possible weighted support (max score 1.0 * total_weight)
            # This makes the score proportional to the total supporting evidence relative to all sources.
            max_possible_support = total_weight
            if max_possible_support > 0:
                final_score = min(1.0, support_sum / max_possible_support)
            else:
                final_score = (
                    0.1  # Fallback if max_possible_support is zero unexpectedly
                )

            # Further adjust: if very little support, push score lower
            if (
                final_score < 0.5 and support_sum < 0.5
            ):  # If score is low and little actual support
                final_score = max(0.1, final_score * 0.8)  # Push down more

            elif (
                final_score > 0.5 and support_sum >= 1.0
            ):  # If score is high and decent support
                final_score = min(0.9, final_score * 1.1)  # Push up more

            # Ensure score is within reasonable bounds
            final_score = max(0.0, min(1.0, final_score))

        result = {
            "score": final_score,
            "total_sources_processed": total_sources_processed,
            "support_sum": support_sum,
            "total_weight": total_weight,
            "source_details": source_details,
        }
        self._add_to_cache(cache_key, result)
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

        sentences = (
            [sent.text for sent in nlp(content).sents] if nlp else content.split(".")
        )

        # We will use the semantic similarity from DistilBERT predominantly
        best_semantic_similarity = 0.0

        for sent in sentences:
            # Calculate semantic similarity using DistilBERT embeddings
            current_similarity = self._semantic_similarity(claim, sent)
            if current_similarity > best_semantic_similarity:
                best_semantic_similarity = current_similarity

        domain_weight, domain_type = self._get_domain_weight(url)

        return best_semantic_similarity, domain_weight, domain_type
