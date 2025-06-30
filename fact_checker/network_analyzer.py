import hashlib
import time
from collections import defaultdict, Counter
from typing import List, Dict
import re

from utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS, extract_domain

SOCIAL_AGGREGATOR_DOMAINS = {
    'facebook.com', 'twitter.com', 'reddit.com', 'youtube.com', 'instagram.com',
    'tiktok.com', 'google.com', 'yahoo.com', 'msn.com', 'aol.com'
}

class NetworkAnalyzer:
    """Analyze propagation patterns and network spread with enhanced detection"""
    
    def __init__(self, max_stories: int = 1000):
        self.story_tracker = defaultdict(lambda: {
            'first_seen': None,
            'domains': set(),
            'timestamps': [],
            'similar_stories': []
        })
        self.max_stories = max_stories
    
    def _generate_story_signature(self, headline: str) -> str:
        """Generate a more robust story signature"""
        # Normalize headline for better matching
        normalized = re.sub(r'[^\w\s]', '', headline.lower())
        normalized = ' '.join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]
    
    def _calculate_domain_credibility_score(self, domains: List[str]) -> float:
        """Calculate overall credibility based on domain mix"""
        if not domains:
            return 0.5
        
        domain_counts = Counter(domains)
        total_sources = len(domains)
        
        trusted_count = sum(count for domain, count in domain_counts.items() 
                          if domain in TRUSTED_DOMAINS)
        suspicious_count = sum(count for domain, count in domain_counts.items() 
                             if domain in SUSPICIOUS_DOMAINS)
        social_count = sum(count for domain, count in domain_counts.items() 
                         if domain in SOCIAL_AGGREGATOR_DOMAINS)
        
        trusted_ratio = trusted_count / total_sources
        suspicious_ratio = suspicious_count / total_sources
        social_ratio = social_count / total_sources
        
        score = 0.5  # Start neutral
        score += trusted_ratio * 0.4  
        score -= suspicious_ratio * 0.5  
        score += social_ratio * 0.1 
        
        return max(0.0, min(1.0, score))
    
    def _detect_coordination_patterns(self, domains: List[str]) -> Dict:
        """Detect potential coordinated inauthentic behavior"""
        domain_counts = Counter(domains)
        unique_domains = len(set(domains))
        total_sources = len(domains)
        
        if unique_domains == 0:
            concentration = 1.0
        else:
            sorted_counts = sorted(domain_counts.values(), reverse=True)
            concentration = sum((2 * i - total_sources + 1) * count 
                              for i, count in enumerate(sorted_counts, 1)) / (total_sources * total_sources)
        
        coordination_flags = []
        
        suspicious_domains = [d for d in domains if d in SUSPICIOUS_DOMAINS]
        if len(suspicious_domains) > len(domains) * 0.6:
            coordination_flags.append("high_suspicious_concentration")
        
        max_domain_share = max(domain_counts.values()) / total_sources if total_sources > 0 else 0
        if max_domain_share > 0.7 and unique_domains > 3:
            coordination_flags.append("single_domain_dominance")
        
        if unique_domains > 20 and total_sources < 30:
            coordination_flags.append("artificial_diversity")
        
        return {
            "concentration_score": concentration,
            "domain_diversity": unique_domains / total_sources if total_sources > 0 else 0,
            "coordination_flags": coordination_flags,
            "max_domain_share": max_domain_share
        }
    
    def analyze_propagation_pattern(self, headline: str, search_results: List[str]) -> Dict:
        """Enhanced propagation analysis with multiple detection methods"""
        story_signature = self._generate_story_signature(headline)
        current_time = time.time()
        
        # Extract and filter domains
        domains = []
        valid_urls = 0
        
        for url in search_results:
            domain = extract_domain(url)
            if domain:
                domains.append(domain)
                valid_urls += 1
        
        if not domains:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "domain_diversity": 0.0,
                "trusted_sources": 0,
                "suspicious_sources": 0,
                "error": "No valid domains found"
            }
        
        # Update story tracking
        story_data = self.story_tracker[story_signature]
        if story_data['first_seen'] is None:
            story_data['first_seen'] = current_time
        
        story_data['domains'].update(domains)
        story_data['timestamps'].append(current_time)
        
        # Analyze domain composition
        domain_stats = Counter(domains)
        unique_domains = len(set(domains))
        
        trusted_count = sum(1 for d in domains if d in TRUSTED_DOMAINS)
        suspicious_count = sum(1 for d in domains if d in SUSPICIOUS_DOMAINS)
        
        # Calculate domain diversity (Shannon entropy-like measure)
        total_domains = len(domains)
        domain_diversity = unique_domains / total_domains if total_domains > 0 else 0
        
        # Enhanced diversity with entropy consideration
        if unique_domains > 1:
            entropy = 0
            for count in domain_stats.values():
                p = count / total_domains
                entropy -= p * (p.log() if hasattr(p, 'log') else 0)  # Simplified entropy
            normalized_entropy = entropy / (unique_domains.bit_length() if unique_domains > 1 else 1)
            domain_diversity = min(domain_diversity + normalized_entropy * 0.2, 1.0)
        
        # Get credibility and coordination scores
        credibility_score = self._calculate_domain_credibility_score(domains)
        coordination_analysis = self._detect_coordination_patterns(domains)
        
        # Calculate final propagation score
        base_score = credibility_score
        
        # Adjust for domain diversity (natural vs artificial spread)
        if domain_diversity > 0.8:
            base_score += 0.15
        elif domain_diversity < 0.2:
            base_score -= 0.2
        
        coordination_penalty = len(coordination_analysis['coordination_flags']) * 0.1
        base_score -= coordination_penalty
        
        if trusted_count >= 3:
            base_score += 0.1
        
        if suspicious_count > trusted_count and suspicious_count > total_domains * 0.4:
            base_score -= 0.2
        
        final_score = max(0.0, min(1.0, base_score))
        
        confidence = min(1.0, (
            min(valid_urls / 10, 1.0) * 0.4 +  
            min(unique_domains / 5, 1.0) * 0.3 +  
            (1 - len(coordination_analysis['coordination_flags']) * 0.1) * 0.3 
        ))
        
        if len(self.story_tracker) > self.max_stories:
            self._cleanup_old_stories()
        
        return {
            "score": final_score,
            "confidence": confidence,
            "domain_diversity": domain_diversity,
            "trusted_sources": trusted_count,
            "suspicious_sources": suspicious_count,
        }
    
    def _cleanup_old_stories(self):
        """Remove oldest stories to prevent memory bloat"""
        current_time = time.time()
        stories_to_remove = []
        
        for story_id, data in self.story_tracker.items():
            # Remove stories older than 7 days
            if current_time - data['first_seen'] > 7 * 24 * 3600:
                stories_to_remove.append(story_id)
        
        for story_id in stories_to_remove:
            del self.story_tracker[story_id]