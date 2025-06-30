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

        # Precompiled suspicious patterns for better performance
        self._compiled_suspicious_patterns = [
            re.compile(r'\d{4,}'),                       # Long numeric domains
            re.compile(r'(fake|hoax|scam|click|bait)', re.IGNORECASE),  # Suspicious keywords
            re.compile(r'[a-z]+\d+[a-z]+', re.IGNORECASE), # Mixed alphanumeric
            re.compile(r'(\.tk|\.ml|\.ga|\.cf)$', re.IGNORECASE) # Suspicious TLDs
        ]

    def analyze_domain_credibility(self, domain: str) -> Dict:
        """Analyze domain credibility using multiple factors"""
        domain = domain.lower()

        if domain in TRUSTED_DOMAINS:
            return {"score": 0.95, "reason": "Verified trusted source"}

        if domain in SUSPICIOUS_DOMAINS:
            return {"score": 0.05, "reason": "Known suspicious source"}

        credibility_score = 0.5  # Start neutral
        reasons = []

        if self._is_suspicious_domain_structure(domain):
            credibility_score -= 0.2
            reasons.append("Suspicious domain structure")

        if self._has_news_indicators(domain):
            credibility_score += 0.1
            reasons.append("Contains news-like keywords")

        tld_score = self._analyze_tld_credibility(domain)
        credibility_score += tld_score

        if tld_score < 0:
            reasons.append("Low-trust TLD")
        elif tld_score > 0:
            reasons.append("High-trust TLD")

        return {
            "score": max(0.0, min(1.0, round(credibility_score, 2))),
            "reason": "; ".join(reasons) if reasons else "Unknown domain"
        }

    def _is_suspicious_domain_structure(self, domain: str) -> bool:
        """Check for suspicious domain patterns"""
        return any(pattern.search(domain) for pattern in self._compiled_suspicious_patterns)

    def _has_news_indicators(self, domain: str) -> bool:
        """Check for legitimate news indicators in domain"""
        news_indicators = ['news', 'times', 'post', 'herald', 'gazette', 'journal', 'tribune']
        return any(indicator in domain for indicator in news_indicators)

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
        # Medium trust is neutral
        return 0.0
