
import hashlib
from collections import defaultdict
from typing import List, Dict

from utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS, extract_domain

class NetworkAnalyzer:
    """Analyze propagation patterns and network spread"""
    
    def __init__(self):
        self.story_tracker = defaultdict(list)
        self.domain_patterns = defaultdict(list)
    
    def analyze_propagation_pattern(self, headline: str, search_results: List[str]) -> Dict:
        """Analyze how a story spreads across different sources"""
        story_hash = hashlib.md5(headline.encode()).hexdigest()[:8]
        
        # Extract domains from search results
        domains = []
        for url in search_results:
            domain = extract_domain(url)
            domains.append(domain)
        
        # Analyze domain diversity
        unique_domains = set(domains)
        domain_diversity = len(unique_domains) / len(domains) if domains else 0
        
        # Check for coordinated spread (same story on multiple suspicious sites)
        suspicious_domain_count = sum(1 for domain in domains if domain in SUSPICIOUS_DOMAINS)
        trusted_domain_count = sum(1 for domain in domains if domain in TRUSTED_DOMAINS)
        
        # Calculate propagation score
        propagation_score = 0.5  # Default
        
        if trusted_domain_count > suspicious_domain_count:
            propagation_score += 0.3
        elif suspicious_domain_count > trusted_domain_count:
            propagation_score -= 0.3
        
        # Bonus for domain diversity (natural spread vs coordinated campaign)
        if domain_diversity > 0.7:
            propagation_score += 0.2
        elif domain_diversity < 0.3:
            propagation_score -= 0.2
        
        return {
            "score": max(0.0, min(1.0, propagation_score)),
            "domain_diversity": domain_diversity,
            "trusted_sources": trusted_domain_count,
            "suspicious_sources": suspicious_domain_count,
            "unique_domains": len(unique_domains)
        }