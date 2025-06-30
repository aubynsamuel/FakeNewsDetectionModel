import re
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("⚠️ Please install spacy English model: python -m spacy download en_core_web_md")
    nlp = None
    
class ClaimExtractor:
    """Extract and verify specific factual claims"""
    
    def __init__(self):
        self.claim_cache = {}
        
    def extract_claims(self, headline: str) -> List[Dict]:
        """Extract verifiable claims from headline"""
        if not nlp:
            return []
        
        doc = nlp(headline)
        claims = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "DATE", "MONEY", "QUANTITY"]:
                claims.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "context": headline,
                    "verifiable": True
                })
        
        # Extract numerical claims
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:percent|%|million|billion|thousand)\b', headline, re.IGNORECASE)
        for number in numbers:
            claims.append({
                "text": number,
                "type": "NUMERICAL",
                "context": headline,
                "verifiable": True
            })
        
        return claims
    
    def verify_claim_against_sources(self, claim: str, search_results: List[str]) -> Dict:
        """Verify a specific claim against search results"""
        verification_score = 0.0
        supporting_sources = 0
        contradicting_sources = 0
        
        for url in search_results[:5]:  # Check top 5 results
            try:
                content = self._extract_content(url)
                if content:
                    # Simple semantic similarity check
                    similarity = self._calculate_semantic_similarity(claim, content)
                    if similarity > 0.7:
                        supporting_sources += 1
                    elif similarity < 0.3:
                        contradicting_sources += 1
            except:
                continue
        
        total_sources = supporting_sources + contradicting_sources
        if total_sources > 0:
            verification_score = supporting_sources / total_sources
        
        return {
            "score": verification_score,
            "supporting": supporting_sources,
            "contradicting": contradicting_sources,
            "confidence": min(total_sources / 3, 1.0)  # More sources = higher confidence
        }
    
    def _extract_content(self, url: str) -> str:
        """Extract content from URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            content = soup.get_text()[:1000]  # First 1000 characters
            return content
        except:
            return ""
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1.lower(), text2.lower()])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0