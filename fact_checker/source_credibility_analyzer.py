import re
from datetime import timedelta
from typing import Dict

from utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS

class SourceCredibilityAnalyzer:
    """Analyzes source credibility beyond simple domain whitelisting"""
    
    def __init__(self):
        self.domain_scores = {}
        self.update_interval = timedelta(hours=24)
        self.last_update = {}
    
    def analyze_domain_credibility(self, domain: str) -> Dict:
        """Analyze domain credibility using multiple factors"""
        if domain in TRUSTED_DOMAINS:
            return {"score": 0.9, "reason": "Verified trusted source"}
        
        if domain in SUSPICIOUS_DOMAINS:
            return {"score": 0.1, "reason": "Known suspicious source"}
        
        # For unknown domains, analyze various factors
        credibility_score = 0.5  # Default neutral score
        reasons = []
        
        # Check domain age and structure
        if self._is_suspicious_domain_structure(domain):
            credibility_score -= 0.2
            reasons.append("Suspicious domain structure")
        
        # Check for news-like indicators
        if self._has_news_indicators(domain):
            credibility_score += 0.1
            reasons.append("Contains news indicators")
        
        # Check TLD credibility
        tld_score = self._analyze_tld_credibility(domain)
        credibility_score += tld_score
        if tld_score < 0:
            reasons.append("Suspicious TLD")
        
        return {
            "score": max(0.0, min(1.0, credibility_score)),
            "reason": "; ".join(reasons) if reasons else "Unknown domain"
        }
    
    def _is_suspicious_domain_structure(self, domain: str) -> bool:
        """Check for suspicious domain patterns"""
        suspicious_patterns = [
            r'\d{4,}',  # Long numbers
            r'(fake|hoax|scam|click|bait)',  # Suspicious words
            r'[a-z]+\d+[a-z]+',  # Mixed letters and numbers
            r'(\.tk|\.ml|\.ga|\.cf)$',  # Suspicious TLDs
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, domain.lower()):
                return True
        return False
    
    def _has_news_indicators(self, domain: str) -> bool:
        """Check for legitimate news indicators"""
        news_indicators = ['news', 'times', 'post', 'herald', 'gazette', 'journal', 'tribune']
        return any(indicator in domain.lower() for indicator in news_indicators)
    
    def _analyze_tld_credibility(self, domain: str) -> float:
        """Analyze TLD credibility"""
        high_trust_tlds = ['.edu', '.gov', '.org']
        medium_trust_tlds = ['.com', '.net', '.co.uk', '.com.au']
        low_trust_tlds = ['.tk', '.ml', '.ga', '.cf', '.info']
        
        for tld in high_trust_tlds:
            if domain.endswith(tld):
                return 0.2
        
        for tld in medium_trust_tlds:
            if domain.endswith(tld):
                return 0.0
        
        for tld in low_trust_tlds:
            if domain.endswith(tld):
                return -0.3
        
        return 0.0