import re
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format=' %(message)s')

# Load SpaCy model once globally with error handling
try:
    nlp = spacy.load("en_core_web_md")  # Using medium model for better word vectors
    logging.info("SpaCy English model 'en_core_web_md' loaded successfully.")
except OSError:
    logging.error("⚠️ SpaCy English model 'en_core_web_md' not found. Please install: python -m spacy download en_core_web_md")
    nlp = None

class SearchEngineRotator:
    """Rotates between different search engines to avoid rate limiting"""
    
    def __init__(self):
        self.engines = ['google', 'bing']
        self.current_index = 0
        self.request_counts = {'google': 0, 'bing': 0}
        self.last_request_time = {'google': 0, 'bing': 0}
        self.rate_limit_delay = 1.0  # seconds between requests
        
    def get_next_engine(self) -> str:
        """Get the next search engine to use"""
        current_time = time.time()
        
        # Check if we need to wait for rate limiting
        for engine in self.engines:
            if current_time - self.last_request_time[engine] < self.rate_limit_delay:
                continue
            return engine
        
        # If all engines are rate-limited, use round-robin
        engine = self.engines[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.engines)
        return engine
    
    def mark_request(self, engine: str):
        """Mark that a request was made to an engine"""
        self.request_counts[engine] += 1
        self.last_request_time[engine] = time.time()

class ClaimExtractor:
    """Extract and verify specific factual claims from headlines/texts with enhanced performance."""
    
    def __init__(self, cache_size: int = 500, max_workers: int = 4):
        self.claim_cache: Dict[str, Dict] = {}
        self.content_cache: Dict[str, str] = {}  # Cache for URL content
        self.cache_size = cache_size
        self.max_workers = max_workers
        self.search_rotator = SearchEngineRotator()
        
        # Optimized headers for different user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        self.current_ua_index = 0
        self.timeout = 10
        
        # Pre-compiled regex patterns for efficiency
        self.number_pattern = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:percent|%|million|billion|thousand|dollars?|\$)\b', re.IGNORECASE)
        self.text_cleanup_pattern = re.compile(r'[^a-z0-9\s]', re.IGNORECASE)
        
        # Initialize TF-IDF vectorizer once
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=0.9,
            max_features=5000,  # Limit features for speed
            ngram_range=(1, 2)  # Include bigrams for better context
        )
        
    def _get_user_agent(self) -> str:
        """Rotate user agents to avoid detection"""
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua
    
    def _cache_key(self, text: str) -> str:
        """Generate a cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, result: Dict):
        """Add result to cache with LRU eviction"""
        if len(self.claim_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.claim_cache))
            del self.claim_cache[oldest_key]
        self.claim_cache[key] = result

    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get result from cache"""
        return self.claim_cache.get(key)

    @lru_cache(maxsize=200)
    def _process_spacy_doc(self, text: str) -> List[Dict]:
        """Cached SpaCy processing for repeated text"""
        if not nlp:
            return []
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            if len(ent.text.strip()) > 2 and ent.label_ in [
                "PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", 
                "DATE", "TIME", "MONEY", "QUANTITY", "PERCENT", "CARDINAL"
            ]:
                entities.append({
                    "text": ent.text.strip(),
                    "type": ent.label_,
                    "start": ent.start,
                    "end": ent.end
                })
        
        return entities

    def extract_claims(self, headline: str) -> List[Dict]:
        """Extract verifiable claims with improved performance"""
        cache_key = self._cache_key(f"extract_{headline}")
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        claims = []
        headline = headline.strip()
        
        if not headline:
            return claims
        
        # 1. Fast regex-based numerical extraction
        numerical_matches = self.number_pattern.findall(headline)
        for match in numerical_matches:
            claims.append({
                "text": match.strip(),
                "type": "NUMERICAL",
                "context": headline,
                "verifiable": True
            })
        
        # 2. SpaCy-based entity extraction (if available)
        if nlp:
            try:
                entities = self._process_spacy_doc(headline)
                for ent in entities:
                    # Avoid duplicates from regex
                    if not any(claim["text"].lower() in ent["text"].lower() for claim in claims):
                        claims.append({
                            "text": ent["text"],
                            "type": ent["type"],
                            "context": headline,
                            "verifiable": True
                        })
                
                # 3. Simple action extraction
                doc = nlp(headline)
                for token in doc:
                    if token.pos_ == "VERB" and token.dep_ == "ROOT":
                        # Extract subject-verb patterns
                        subjects = [child for child in token.children if child.dep_ in ["nsubj", "nsubjpass"]]
                        if subjects:
                            subject_text = " ".join([subj.text for subj in subjects])
                            action_claim = f"{subject_text} {token.text}"
                            
                            # Only add if it contains named entities
                            if any(ent["text"].lower() in action_claim.lower() for ent in entities):
                                claims.append({
                                    "text": action_claim,
                                    "type": "ACTION_CLAIM",
                                    "context": headline,
                                    "verifiable": True
                                })
            except Exception as e:
                logging.debug(f"SpaCy processing error: {e}")
        
        # Remove duplicates and filter short claims
        unique_claims = []
        seen_texts = set()
        for claim in claims:
            claim_text = claim["text"].lower().strip()
            if len(claim_text) > 3 and claim_text not in seen_texts:
                unique_claims.append(claim)
                seen_texts.add(claim_text)
        
        # Cache result
        self._add_to_cache(cache_key, unique_claims)
        return unique_claims

    def verify_claim_against_sources(self, claim: str, search_results: List[str]) -> Dict:
        """Verify a specific claim against search results using enhanced similarity methods"""
        cache_key = self._cache_key(f"verify_{claim}")
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logging.debug(f"Claim verification found in cache: '{claim[:50]}...'")
            return cached_result

        verification_score = 0.0
        supporting_sources = 0
        contradicting_sources = 0
        total_content_sources = 0
        
        # Process URLs in parallel for speed
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self._extract_content, url): url 
                for url in search_results[:8]  # Process top 8 results
            }
            
            for future in as_completed(future_to_url, timeout=30):
                url = future_to_url[future]
                try:
                    content = future.result(timeout=10)
                    if content and len(content.strip()) > 50:  # Minimum content threshold
                        total_content_sources += 1
                        
                        # Use SpaCy's word vectors for semantic similarity if available
                        similarity = self._calculate_enhanced_similarity(claim, content)
                        
                        logging.debug(f"Claim similarity with {url}: {similarity:.3f}")
                        
                        # Enhanced thresholds based on en_core_web_md capabilities
                        if similarity > 0.8:  # Very strong support
                            supporting_sources += 2  # Weight stronger evidence more
                        elif similarity > 0.65:  # Strong support
                            supporting_sources += 1
                        elif similarity < 0.2:  # Strong contradiction
                            contradicting_sources += 1
                        elif similarity < 0.1:  # Very strong contradiction
                            contradicting_sources += 2
                        
                except Exception as e:
                    logging.debug(f"Error processing {url}: {e}")
                    continue
        
        # Calculate verification score with weighted evidence
        total_evidence = supporting_sources + contradicting_sources
        if total_evidence > 0:
            verification_score = supporting_sources / total_evidence
        else:
            verification_score = 0.5  # Neutral if no strong evidence
        
        # Enhanced confidence calculation
        confidence_score = min(total_content_sources / 4.0, 1.0)
        
        # Boost confidence if we have high-quality word vector similarities
        if nlp and nlp(claim).has_vector:
            confidence_score = min(confidence_score * 1.2, 1.0)
        
        result = {
            "score": verification_score,
            "supporting_sources_count": supporting_sources,
            "contradicting_sources_count": contradicting_sources,
            "confidence": confidence_score,
            "total_sources_processed": total_content_sources
        }
        
        self._add_to_cache(cache_key, result)
        return result
    
    def _calculate_enhanced_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using SpaCy's word vectors (en_core_web_md benefits)
        with TF-IDF fallback for robustness
        """
        try:
            # Primary method: Use SpaCy's word vectors for semantic similarity
            if nlp and nlp(text1).has_vector and nlp(text2).has_vector:
                doc1 = nlp(text1.lower())
                doc2 = nlp(text2.lower())
                
                # SpaCy's built-in similarity uses cosine similarity of averaged word vectors
                spacy_similarity = doc1.similarity(doc2)
                
                # Enhanced similarity with entity matching bonus
                entity_bonus = self._calculate_entity_overlap(doc1, doc2)
                
                # Combine similarities with weights
                final_similarity = (spacy_similarity * 0.8) + (entity_bonus * 0.2)
                
                logging.debug(f"SpaCy similarity: {spacy_similarity:.3f}, Entity bonus: {entity_bonus:.3f}")
                return min(final_similarity, 1.0)
            
            # Fallback: Enhanced TF-IDF similarity
            return self._calculate_tfidf_similarity(text1, text2)
            
        except Exception as e:
            logging.debug(f"Error in enhanced similarity calculation: {e}")
            return self._calculate_tfidf_similarity(text1, text2)
    
    def _calculate_entity_overlap(self, doc1, doc2) -> float:
        """Calculate bonus score based on overlapping named entities"""
        try:
            entities1 = {ent.text.lower() for ent in doc1.ents if ent.label_ in [
                "PERSON", "ORG", "GPE", "EVENT", "DATE", "MONEY", "QUANTITY"
            ]}
            entities2 = {ent.text.lower() for ent in doc2.ents if ent.label_ in [
                "PERSON", "ORG", "GPE", "EVENT", "DATE", "MONEY", "QUANTITY"
            ]}
            
            if not entities1 or not entities2:
                return 0.0
            
            overlap = entities1.intersection(entities2)
            total_unique = entities1.union(entities2)
            
            return len(overlap) / len(total_unique) if total_unique else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Enhanced TF-IDF similarity calculation with preprocessing"""
        try:
            # Preprocess texts
            processed_text1 = self.text_cleanup_pattern.sub(' ', text1.lower()).strip()
            processed_text2 = self.text_cleanup_pattern.sub(' ', text2.lower()).strip()
            
            if not processed_text1 or not processed_text2:
                return 0.0
            
            # Use class vectorizer for consistency
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text1, processed_text2])
            except ValueError:
                # Fallback to simple vectorizer if class vectorizer fails
                simple_vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
                tfidf_matrix = simple_vectorizer.fit_transform([processed_text1, processed_text2])
            
            if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
                return 0.0
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
            
        except Exception as e:
            logging.debug(f"TF-IDF similarity error: {e}")
            return 0.0

    def _extract_content(self, url: str) -> str:
        """
        Enhanced content extraction with caching and robust parsing
        """
        # Check content cache first
        cache_key = self._cache_key(url)
        if cache_key in self.content_cache:
            return self.content_cache[cache_key]
        
        try:
            headers = {'User-Agent': self._get_user_agent()}
            
            # Add delay to avoid being blocked
            time.sleep(0.5)
            
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove irrelevant content
            for element in soup(["script", "style", "header", "footer", "nav", "aside", "form", "iframe"]):
                element.decompose()
            
            # Prioritize main content areas
            content_selectors = [
                'article', 'main', '[role="main"]', '.content', '.article-content',
                '.post-content', '.entry-content', '.article-body'
            ]
            
            extracted_text = ""
            
            # Try to find main content
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    extracted_text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                    break
            
            # Fallback to all paragraphs and divs
            if not extracted_text:
                content_elements = soup.find_all(['p', 'div'], class_=lambda x: x is None or 'ad' not in str(x).lower())
                extracted_text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in content_elements])
            
            # Final fallback
            if not extracted_text:
                extracted_text = soup.get_text(separator=' ', strip=True)
            
            # Clean and limit content
            content = re.sub(r'\s+', ' ', extracted_text).strip()
            content = content[:8000]  # Increased limit for better context with en_core_web_md
            
            # Cache the result
            if len(self.content_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.content_cache))
                del self.content_cache[oldest_key]
            
            self.content_cache[cache_key] = content
            return content
            
        except requests.exceptions.Timeout:
            logging.debug(f"Request timeout for {url}")
            return ""
        except requests.exceptions.RequestException as e:
            logging.debug(f"Request error for {url}: {e}")
            return ""
        except Exception as e:
            logging.debug(f"Content extraction error for {url}: {e}")
            return ""
