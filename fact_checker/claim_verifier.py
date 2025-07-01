import re
from typing import List, Dict, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from urllib.parse import urlparse
from utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS

# Configure logging
logging.basicConfig(level=logging.INFO, format=' %(message)s')

# Load SpaCy model once globally with error handling
try:
    nlp = spacy.load("en_core_web_md")
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
        self.rate_limit_delay = 1.0
        
    def get_next_engine(self) -> str:
        """Get the next search engine to use"""
        current_time = time.time()
        
        for engine in self.engines:
            if current_time - self.last_request_time[engine] < self.rate_limit_delay:
                continue
            return engine
        
        engine = self.engines[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.engines)
        return engine
    
    def mark_request(self, engine: str):
        """Mark that a request was made to an engine"""
        self.request_counts[engine] += 1
        self.last_request_time[engine] = time.time()

class ClaimVerifier:
    """Extract and verify specific factual claims from headlines/texts with enhanced accuracy and domain weighting."""
    
    def __init__(self, cache_size: int = 500, max_workers: int = 4):
        self.claim_cache: Dict[str, Dict] = {}
        self.content_cache: Dict[str, str] = {}
        self.cache_size = cache_size
        self.max_workers = max_workers
        self.search_rotator = SearchEngineRotator()
        
        # Domain weighting system
        self.trusted_domains = TRUSTED_DOMAINS
        self.suspicious_domains = SUSPICIOUS_DOMAINS
        
        # Weights for different domain types
        self.domain_weights = {
            'trusted': 2.0,      # Trusted sources get double weight
            'suspicious': 0.3,   # Suspicious sources get reduced weight
            'neutral': 1.0       # Unknown domains get normal weight
        }
        
        # Enhanced similarity thresholds
        self.similarity_thresholds = {
            'very_strong_support': 0.75,    # Lowered from 0.8
            'strong_support': 0.55,         # Lowered from 0.65
            'weak_support': 0.35,           # New threshold
            'neutral': 0.25,                # New threshold
            'weak_contradiction': 0.15,     # New threshold
            'strong_contradiction': 0.1     # Raised from 0.1
        }
        
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        self.current_ua_index = 0
        self.timeout = 10
        
        # Pre-compiled patterns
        self.number_pattern = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:percent|%|million|billion|thousand|dollars?|\$|GHS|cedi)\b', re.IGNORECASE)
        self.text_cleanup_pattern = re.compile(r'[^a-z0-9\s]', re.IGNORECASE)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=0.9,
            max_features=5000,
            ngram_range=(1, 2)
        )
        
    def _get_domain_weight(self, url: str) -> Tuple[float, str]:
        """Get the credibility weight for a domain"""
        try:
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix
            domain = domain.replace('www.', '')
            
            if domain in self.trusted_domains:
                return self.domain_weights['trusted'], 'trusted'
            elif domain in self.suspicious_domains:
                return self.domain_weights['suspicious'], 'suspicious'
            else:
                return self.domain_weights['neutral'], 'neutral'
        except Exception:
            return self.domain_weights['neutral'], 'neutral'
    
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
            oldest_key = next(iter(self.claim_cache))
            del self.claim_cache[oldest_key]
        self.claim_cache[key] = result

    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get result from cache"""
        return self.claim_cache.get(key)

    def verify_claim_against_sources(self, claim: str, search_results: List[str]) -> Dict:
        """Verify a specific claim against search results with domain weighting and improved accuracy"""
        cache_key = self._cache_key(f"verify_{claim}")
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logging.debug(f"Claim verification found in cache: '{claim[:50]}...'")
            return cached_result

        # Enhanced scoring system
        weighted_support_score = 0.0
        weighted_contradiction_score = 0.0
        total_weight = 0.0
        
        source_details = []
        total_sources_processed = 0
        
        # Process URLs in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self._extract_and_analyze_content, url, claim): url 
                for url in search_results[:12]  # Process more sources for better accuracy
            }
            
            for future in as_completed(future_to_url, timeout=45):
                url = future_to_url[future]
                try:
                    result = future.result(timeout=15)
                    if result:
                        content, similarity, domain_weight, domain_type = result
                        total_sources_processed += 1
                        total_weight += domain_weight
                        
                        # Enhanced scoring based on similarity and domain credibility
                        evidence_score = self._calculate_evidence_score(similarity, domain_weight)
                        
                        # Determine if this is supporting or contradicting evidence
                        if similarity >= self.similarity_thresholds['weak_support']:
                            weighted_support_score += evidence_score
                        elif similarity <= self.similarity_thresholds['weak_contradiction']:
                            weighted_contradiction_score += evidence_score
                        
                        source_details.append({
                            'url': url,
                            'similarity': similarity,
                            'domain_weight': domain_weight,
                            'domain_type': domain_type,
                            'evidence_score': evidence_score
                        })
                        
                        logging.debug(f"URL: {url[:50]}... | Similarity: {similarity:.3f} | Weight: {domain_weight} | Type: {domain_type}")
                        
                except Exception as e:
                    logging.debug(f"Error processing {url}: {e}")
                    continue
        
        # Calculate final verification score with improved logic
        verification_score = self._calculate_final_score(
            weighted_support_score, 
            weighted_contradiction_score, 
            total_weight,
            total_sources_processed
        )
        
        # Enhanced confidence calculation
        confidence_score = self._calculate_confidence(
            total_sources_processed, 
            total_weight, 
            source_details
        )
        
        # Add penalty for suspicious sources dominating
        suspicious_penalty = self._calculate_suspicious_penalty(source_details)
        verification_score = max(0.0, verification_score - suspicious_penalty)
        
        result = {
            "score": verification_score,
            "confidence": confidence_score,
            "total_sources_processed": total_sources_processed,
            "weighted_support_score": weighted_support_score,
            "weighted_contradiction_score": weighted_contradiction_score,
            "total_weight": total_weight,
            "source_breakdown": {
                'trusted': len([s for s in source_details if s['domain_type'] == 'trusted']),
                'suspicious': len([s for s in source_details if s['domain_type'] == 'suspicious']),
                'neutral': len([s for s in source_details if s['domain_type'] == 'neutral'])
            },
            "suspicious_penalty": suspicious_penalty
        }
        
        self._add_to_cache(cache_key, result)
        return result
    
    def _extract_and_analyze_content(self, url: str, claim: str) -> Optional[Tuple[str, float, float, str]]:
        """Extract content and analyze similarity with domain weighting"""
        content = self._extract_content(url)
        if not content or len(content.strip()) < 50:
            return None
        
        similarity = self._calculate_enhanced_similarity(claim, content)
        domain_weight, domain_type = self._get_domain_weight(url)
        
        return content, similarity, domain_weight, domain_type
    
    def _calculate_evidence_score(self, similarity: float, domain_weight: float) -> float:
        """Calculate weighted evidence score"""
        # Base evidence strength
        if similarity >= self.similarity_thresholds['very_strong_support']:
            base_score = 3.0
        elif similarity >= self.similarity_thresholds['strong_support']:
            base_score = 2.0
        elif similarity >= self.similarity_thresholds['weak_support']:
            base_score = 1.0
        elif similarity <= self.similarity_thresholds['strong_contradiction']:
            base_score = 3.0  # Strong contradiction is also strong evidence
        elif similarity <= self.similarity_thresholds['weak_contradiction']:
            base_score = 1.5
        else:
            base_score = 0.5  # Neutral/weak evidence
        
        return base_score * domain_weight
    
    def _calculate_final_score(self, support_score: float, contradiction_score: float, 
                             total_weight: float, total_sources: int) -> float:
        """Calculate final verification score with improved logic"""
        if total_sources == 0 or total_weight == 0:
            return 0.1  # Very low confidence for no evidence
        
        # If we have very little evidence, lean towards skepticism
        if total_sources < 3:
            return 0.2
        
        total_evidence = support_score + contradiction_score
        
        if total_evidence == 0:
            return 0.3  # Slight skepticism for no clear evidence
        
        # Calculate ratio but with more conservative approach
        support_ratio = support_score / total_evidence
        
        # Apply stricter thresholds for claiming something is true
        if support_ratio >= 0.8 and support_score >= 4.0:  # Need strong evidence
            return min(0.9, support_ratio)  # Cap at 0.9 to show some uncertainty
        elif support_ratio >= 0.7 and support_score >= 2.0:
            return min(0.7, support_ratio)
        elif support_ratio >= 0.6:
            return min(0.5, support_ratio)
        else:
            # More likely to be false or unverified
            return max(0.1, support_ratio * 0.5)
    
    def _calculate_confidence(self, total_sources: int, total_weight: float, 
                            source_details: List[Dict]) -> float:
        """Calculate confidence score based on source quality and quantity"""
        if total_sources == 0:
            return 0.0
        
        # Base confidence from source count
        source_confidence = min(total_sources / 8.0, 1.0)
        
        # Boost for trusted sources
        trusted_sources = len([s for s in source_details if s['domain_type'] == 'trusted'])
        trusted_boost = min(trusted_sources / 4.0, 0.3)
        
        # Penalty for suspicious sources
        suspicious_sources = len([s for s in source_details if s['domain_type'] == 'suspicious'])
        suspicious_penalty = min(suspicious_sources / total_sources * 0.4, 0.4)
        
        # Weight quality bonus
        avg_weight = total_weight / total_sources if total_sources > 0 else 1.0
        weight_bonus = min((avg_weight - 1.0) * 0.2, 0.2)
        
        final_confidence = source_confidence + trusted_boost + weight_bonus - suspicious_penalty
        return max(0.1, min(1.0, final_confidence))
    
    def _calculate_suspicious_penalty(self, source_details: List[Dict]) -> float:
        """Calculate penalty if suspicious sources dominate"""
        if not source_details:
            return 0.0
        
        suspicious_count = len([s for s in source_details if s['domain_type'] == 'suspicious'])
        total_count = len(source_details)
        
        if suspicious_count / total_count > 0.5:  # More than half are suspicious
            return 0.3
        elif suspicious_count / total_count > 0.3:  # More than 30% are suspicious
            return 0.15
        else:
            return 0.0
    
    def _calculate_enhanced_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity with improved accuracy"""
        try:
            # Look for explicit contradictions first
            contradiction_score = self._check_explicit_contradictions(text1, text2)
            if contradiction_score > 0:
                return max(0.05, 0.3 - contradiction_score)  # Strong contradiction
            
            # Primary method: SpaCy similarity
            if nlp and nlp(text1).has_vector and nlp(text2).has_vector:
                doc1 = nlp(text1.lower())
                doc2 = nlp(text2.lower())
                
                spacy_similarity = doc1.similarity(doc2)
                entity_bonus = self._calculate_entity_overlap(doc1, doc2)
                
                # Enhanced similarity with better weighting
                final_similarity = (spacy_similarity * 0.7) + (entity_bonus * 0.3)
                
                # Check for semantic contradictions in SpaCy results
                if self._contains_negation_context(text2, text1):
                    final_similarity = max(0.05, final_similarity * 0.3)
                
                return min(final_similarity, 1.0)
            
            return self._calculate_tfidf_similarity(text1, text2)
            
        except Exception as e:
            logging.debug(f"Error in enhanced similarity calculation: {e}")
            return self._calculate_tfidf_similarity(text1, text2)
    
    def _check_explicit_contradictions(self, claim: str, content: str) -> float:
        """Check for explicit contradictions in the content"""
        claim_lower = claim.lower()
        content_lower = content.lower()
        
        # Patterns that might indicate contradiction
        contradiction_patterns = [
            r'\b(no|not|never|false|fake|untrue|debunk|myth|hoax)\b.*' + re.escape(claim_lower.split()[0:3]),
            r'\b(deny|denies|denied|refute|refutes|dispute|disputes)\b',
            r'\b(incorrect|wrong|inaccurate|misleading)\b'
        ]
        
        contradiction_score = 0.0
        for pattern in contradiction_patterns:
            if re.search(pattern, content_lower):
                contradiction_score += 0.3
        
        return min(contradiction_score, 1.0)
    
    def _contains_negation_context(self, content: str, claim: str) -> bool:
        """Check if content contains negation in context of the claim"""
        content_lower = content.lower()
        claim_words = claim.lower().split()[:5]  # First 5 words of claim
        
        for word in claim_words:
            if len(word) > 3:  # Skip short words
                # Look for negation near claim words
                pattern = r'\b(no|not|never|false|deny|denies|wrong|incorrect|untrue|fake|hoax|myth)\b.*\b' + re.escape(word) + r'\b'
                if re.search(pattern, content_lower):
                    return True
                    
                # Reverse pattern
                pattern = r'\b' + re.escape(word) + r'\b.*\b(no|not|never|false|denied|wrong|incorrect|untrue|fake|hoax|myth)\b'
                if re.search(pattern, content_lower):
                    return True
        
        return False
    
    def _calculate_entity_overlap(self, doc1, doc2) -> float:
        """Calculate bonus score based on overlapping named entities"""
        try:
            entities1 = {ent.text.lower() for ent in doc1.ents if ent.label_ in [
                "PERSON", "ORG", "GPE", "EVENT", "DATE", "MONEY", "QUANTITY", "LAW"
            ]}
            entities2 = {ent.text.lower() for ent in doc2.ents if ent.label_ in [
                "PERSON", "ORG", "GPE", "EVENT", "DATE", "MONEY", "QUANTITY", "LAW"
            ]}
            
            if not entities1 or not entities2:
                return 0.0
            
            overlap = entities1.intersection(entities2)
            total_unique = entities1.union(entities2)
            
            return len(overlap) / len(total_unique) if total_unique else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Enhanced TF-IDF similarity calculation"""
        try:
            processed_text1 = self.text_cleanup_pattern.sub(' ', text1.lower()).strip()
            processed_text2 = self.text_cleanup_pattern.sub(' ', text2.lower()).strip()
            
            if not processed_text1 or not processed_text2:
                return 0.0
            
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text1, processed_text2])
            except ValueError:
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
        """Enhanced content extraction with proper character encoding handling"""
        cache_key = self._cache_key(url)
        if cache_key in self.content_cache:
            return self.content_cache[cache_key]
        
        try:
            headers = {
                'User-Agent': self._get_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            time.sleep(0.5)
            
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Handle encoding properly
            if response.encoding is None or response.encoding.lower() in ['iso-8859-1', 'ascii']:
                response.encoding = 'utf-8'
            
            try:
                html_content = response.text
            except UnicodeDecodeError:
                try:
                    html_content = response.content.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    html_content = response.content.decode('latin-1', errors='replace')
            
            # Clean problematic characters
            html_content = html_content.replace('\ufffd', ' ')
            html_content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', ' ', html_content)
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove irrelevant content
            for element in soup(["script", "style", "header", "footer", "nav", "aside", "form", "iframe"]):
                element.decompose()
            
            # Extract content
            content_selectors = [
                'article', 'main', '[role="main"]', '.content', '.article-content',
                '.post-content', '.entry-content', '.article-body'
            ]
            
            extracted_text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    extracted_text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                    break
            
            if not extracted_text:
                content_elements = soup.find_all(['p', 'div'], class_=lambda x: x is None or 'ad' not in str(x).lower())
                extracted_text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in content_elements])
            
            if not extracted_text:
                extracted_text = soup.get_text(separator=' ', strip=True)
            
            # Normalize and clean
            try:
                import unicodedata
                extracted_text = unicodedata.normalize('NFKD', extracted_text)
            except:
                pass
            
            content = re.sub(r'\s+', ' ', extracted_text).strip()
            content = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF]', ' ', content)
            content = content[:8000]
            
            # Cache result
            if len(self.content_cache) >= self.cache_size:
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