import hashlib
import time
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import re

from utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS, extract_domain

SOCIAL_AGGREGATOR_DOMAINS = {
    'facebook.com', 'twitter.com', 'reddit.com', 'youtube.com', 'instagram.com',
    'tiktok.com', 'google.com', 'yahoo.com', 'msn.com', 'aol.com', 'linkedin.com',
    'pinterest.com', 'snapchat.com', 'discord.com', 'telegram.org'
}

# Known content farms and low-quality aggregators
CONTENT_FARM_DOMAINS = {
    'buzzfeed.com', 'clickhole.com', 'upworthy.com', 'viralthread.com',
    'shareably.net', 'littlethings.com', 'providr.com', 'shared.com'
}

class NetworkAnalyzer:
    """Enhanced propagation pattern analyzer with improved scoring accuracy"""
    
    def __init__(self, max_stories: int = 1000):
        self.story_tracker = defaultdict(lambda: {
            'first_seen': None,
            'domains': set(),
            'timestamps': [],
            'similar_stories': [],
            'propagation_velocity': 0
        })
        self.max_stories = max_stories
        
        # Enhanced scoring weights
        self.weights = {
            'domain_credibility': 0.35,    # Primary factor
            'diversity_quality': 0.25,     # Natural vs artificial spread
            'coordination_penalty': 0.20,  # Detect manipulation
            'velocity_analysis': 0.15,     # Propagation speed patterns
            'source_quality': 0.05        # Additional quality factors
        }
        
        # Minimum thresholds for reliable analysis
        self.min_sources_threshold = 3
        self.min_unique_domains = 2
    
    def _generate_story_signature(self, headline: str) -> str:
        """Generate a more robust story signature with better normalization"""
        # Enhanced normalization
        normalized = re.sub(r'[^\w\s]', '', headline.lower())
        # Remove common stop words that don't affect story identity
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [w for w in normalized.split() if w not in stop_words and len(w) > 2]
        normalized = ' '.join(words)
        return hashlib.md5(normalized.encode()).hexdigest()[:12]
    
    def _calculate_domain_credibility_score(self, domains: List[str]) -> Dict:
        """Enhanced domain credibility calculation with detailed breakdown"""
        if not domains:
            return {"score": 0.0, "breakdown": {}, "flags": ["no_domains"]}
        
        domain_counts = Counter(domains)
        total_sources = len(domains)
        unique_domains = len(set(domains))
        
        # Categorize domains
        trusted_count = sum(count for domain, count in domain_counts.items() 
                          if domain in TRUSTED_DOMAINS)
        suspicious_count = sum(count for domain, count in domain_counts.items() 
                             if domain in SUSPICIOUS_DOMAINS)
        social_count = sum(count for domain, count in domain_counts.items() 
                         if domain in SOCIAL_AGGREGATOR_DOMAINS)
        content_farm_count = sum(count for domain, count in domain_counts.items() 
                               if domain in CONTENT_FARM_DOMAINS)
        
        # Calculate ratios
        trusted_ratio = trusted_count / total_sources
        suspicious_ratio = suspicious_count / total_sources
        social_ratio = social_count / total_sources
        content_farm_ratio = content_farm_count / total_sources
        unknown_ratio = 1 - (trusted_ratio + suspicious_ratio + social_ratio + content_farm_ratio)
        
        # Start with low base score for unknown content
        base_score = 0.15
        
        # Apply weighted adjustments
        score = base_score
        score += trusted_ratio * 0.6      # Strong boost for trusted sources
        score -= suspicious_ratio * 0.8   # Heavy penalty for suspicious sources
        score -= content_farm_ratio * 0.4 # Penalty for content farms
        score += social_ratio * 0.1       # Small boost for social verification
        score -= unknown_ratio * 0.2      # Penalty for unknown sources
        
        # Additional quality factors
        flags = []
        
        # Flag if dominated by suspicious sources
        if suspicious_ratio > 0.5:
            score -= 0.3
            flags.append("suspicious_dominance")
            
        # Flag if no trusted sources at all
        if trusted_count == 0 and total_sources > 5:
            score -= 0.2
            flags.append("no_trusted_sources")
            
        # Flag content farm dominance
        if content_farm_ratio > 0.4:
            score -= 0.15
            flags.append("content_farm_dominance")
        
        breakdown = {
            "trusted_ratio": trusted_ratio,
            "suspicious_ratio": suspicious_ratio,
            "social_ratio": social_ratio,
            "content_farm_ratio": content_farm_ratio,
            "unknown_ratio": unknown_ratio,
            "unique_domains": unique_domains,
            "total_sources": total_sources
        }
        
        return {
            "score": max(0.0, min(1.0, score)),
            "breakdown": breakdown,
            "flags": flags
        }
    
    def _calculate_diversity_quality(self, domains: List[str]) -> Dict:
        """Calculate diversity quality - distinguishing natural from artificial spread"""
        if len(domains) < 2:
            return {"score": 0.0, "entropy": 0.0, "flags": ["insufficient_sources"]}
        
        domain_counts = Counter(domains)
        unique_domains = len(set(domains))
        total_sources = len(domains)
        
        # Calculate Shannon entropy for true diversity measurement
        entropy = 0.0
        for count in domain_counts.values():
            p = count / total_sources
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalize entropy (0-1 scale)
        max_entropy = math.log2(unique_domains) if unique_domains > 1 else 0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Base diversity score
        diversity_score = normalized_entropy
        
        flags = []
        
        # Detect artificial patterns
        max_domain_share = max(domain_counts.values()) / total_sources
        
        # Single domain dominance (artificial amplification)
        if max_domain_share > 0.7 and unique_domains > 3:
            diversity_score -= 0.4
            flags.append("single_domain_dominance")
        
        # Artificial diversity (too many unique domains with single mentions)
        single_mention_domains = sum(1 for count in domain_counts.values() if count == 1)
        if single_mention_domains > total_sources * 0.8 and total_sources > 10:
            diversity_score -= 0.3
            flags.append("artificial_diversity")
        
        # Healthy diversity bonus
        if 0.3 <= normalized_entropy <= 0.8 and unique_domains >= 3:
            diversity_score += 0.2
            flags.append("healthy_diversity")
        
        return {
            "score": max(0.0, min(1.0, diversity_score)),
            "entropy": normalized_entropy,
            "max_domain_share": max_domain_share,
            "flags": flags
        }
    
    def _detect_coordination_patterns(self, domains: List[str], timestamps: List[float]) -> Dict:
        """Enhanced coordination pattern detection"""
        domain_counts = Counter(domains)
        unique_domains = len(set(domains))
        total_sources = len(domains)
        
        coordination_score = 1.0  # Start high, reduce for suspicious patterns
        flags = []
        
        if total_sources == 0:
            return {"score": 0.0, "flags": ["no_data"]}
        
        # Gini coefficient for domain concentration
        if unique_domains > 1:
            sorted_counts = sorted(domain_counts.values())
            n = len(sorted_counts)
            cumsum = sum((2 * i - n + 1) * count for i, count in enumerate(sorted_counts, 1))
            gini = cumsum / (n * sum(sorted_counts))
            
            # High concentration penalty
            if gini > 0.7:
                coordination_score -= 0.4
                flags.append("high_domain_concentration")
        
        # Suspicious domain concentration
        suspicious_domains = [d for d in domains if d in SUSPICIOUS_DOMAINS]
        if len(suspicious_domains) > total_sources * 0.6:
            coordination_score -= 0.5
            flags.append("suspicious_coordination")
        
        # Temporal analysis if timestamps available
        if len(timestamps) > 3:
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_time_diff = sum(time_diffs) / len(time_diffs)
            
            # Suspiciously uniform timing
            if all(abs(diff - avg_time_diff) < avg_time_diff * 0.1 for diff in time_diffs):
                coordination_score -= 0.3
                flags.append("uniform_timing_pattern")
        
        # Artificial amplification patterns
        if unique_domains < 3 and total_sources > 10:
            coordination_score -= 0.3
            flags.append("artificial_amplification")
        
        return {
            "score": max(0.0, coordination_score),
            "flags": flags,
            "gini_coefficient": gini if 'gini' in locals() else 0
        }
    
    def _analyze_propagation_velocity(self, timestamps: List[float]) -> Dict:
        """Analyze propagation velocity patterns"""
        if len(timestamps) < 2:
            return {"score": 0.5, "velocity": 0, "flags": ["insufficient_data"]}
        
        sorted_times = sorted(timestamps)
        total_time_span = sorted_times[-1] - sorted_times[0]
        velocity_score = 0.5  # Neutral start
        flags = []
        
        if total_time_span > 0:
            # Sources per hour
            velocity = len(timestamps) / (total_time_span / 3600)
            
            # Natural propagation patterns get higher scores
            if 0.5 <= velocity <= 5:  # Natural spread rate
                velocity_score += 0.2
                flags.append("natural_velocity")
            elif velocity > 20:  # Suspiciously fast
                velocity_score -= 0.3
                flags.append("suspicious_velocity")
            elif velocity < 0.1:  # Suspiciously slow
                velocity_score -= 0.1
                flags.append("slow_propagation")
        
        return {
            "score": max(0.0, min(1.0, velocity_score)),
            "velocity": velocity if 'velocity' in locals() else 0,
            "time_span_hours": total_time_span / 3600 if total_time_span > 0 else 0,
            "flags": flags
        }
    
    def analyze_propagation_pattern(self, headline: str, search_results: List[str]) -> Dict:
        """Enhanced propagation analysis with comprehensive scoring"""
        story_signature = self._generate_story_signature(headline)
        current_time = time.time()
        
        # Extract and validate domains
        domains = []
        valid_urls = 0
        
        for url in search_results:
            domain = extract_domain(url)
            if domain and domain not in ['', 'localhost']:  # Filter invalid domains
                domains.append(domain)
                valid_urls += 1
        
        # Early return for insufficient data
        if len(domains) < self.min_sources_threshold:
            return {
                "score": 0.1,  # Very low score for insufficient data
                "confidence": 0.0,
                "domain_diversity": 0.0,
                "trusted_sources": 0,
                "suspicious_sources": 0,
                "analysis": "Insufficient sources for reliable analysis",
                "flags": ["insufficient_sources"]
            }
        
        # Update story tracking
        story_data = self.story_tracker[story_signature]
        if story_data['first_seen'] is None:
            story_data['first_seen'] = current_time
        
        story_data['domains'].update(domains)
        story_data['timestamps'].append(current_time)
        
        # Perform comprehensive analysis
        credibility_analysis = self._calculate_domain_credibility_score(domains)
        diversity_analysis = self._calculate_diversity_quality(domains)
        coordination_analysis = self._detect_coordination_patterns(domains, story_data['timestamps'])
        velocity_analysis = self._analyze_propagation_velocity(story_data['timestamps'])
        
        # Calculate weighted final score
        final_score = (
            credibility_analysis['score'] * self.weights['domain_credibility'] +
            diversity_analysis['score'] * self.weights['diversity_quality'] +
            coordination_analysis['score'] * self.weights['coordination_penalty'] +
            velocity_analysis['score'] * self.weights['velocity_analysis']
        )
        
        # Additional quality checks
        unique_domains = len(set(domains))
        trusted_count = sum(1 for d in domains if d in TRUSTED_DOMAINS)
        suspicious_count = sum(1 for d in domains if d in SUSPICIOUS_DOMAINS)
        
        # Source quality bonus/penalty
        if trusted_count >= 3 and suspicious_count == 0:
            final_score += 0.1
        elif suspicious_count > trusted_count:
            final_score -= 0.15
        
        # Ensure minimum quality threshold
        if unique_domains < self.min_unique_domains:
            final_score = min(final_score, 0.3)
        
        final_score = max(0.0, min(1.0, final_score))
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(
            valid_urls, unique_domains, credibility_analysis, 
            diversity_analysis, coordination_analysis
        )
        
        # Compile all flags
        all_flags = (credibility_analysis['flags'] + diversity_analysis['flags'] + 
                    coordination_analysis['flags'] + velocity_analysis['flags'])
        
        # Cleanup old stories periodically
        if len(self.story_tracker) > self.max_stories:
            self._cleanup_old_stories()
        
        return {
            "score": round(final_score, 3),
            "confidence": round(confidence, 3),
            "domain_diversity": round(diversity_analysis['entropy'], 3),
            "trusted_sources": trusted_count,
            "suspicious_sources": suspicious_count,
            "unique_domains": unique_domains,
            "total_sources": len(domains),
            "analysis_breakdown": {
                "credibility": credibility_analysis,
                "diversity": diversity_analysis,
                "coordination": coordination_analysis,
                "velocity": velocity_analysis
            },
            "flags": list(set(all_flags)),  # Remove duplicates
            "risk_assessment": self._assess_risk_level(final_score, all_flags)
        }
    
    def _calculate_confidence(self, valid_urls: int, unique_domains: int, 
                            credibility_analysis: Dict, diversity_analysis: Dict,
                            coordination_analysis: Dict) -> float:
        """Calculate confidence in the analysis"""
        confidence = 0.0
        
        # Sample size confidence
        confidence += min(valid_urls / 15, 1.0) * 0.3
        
        # Domain diversity confidence  
        confidence += min(unique_domains / 8, 1.0) * 0.25
        
        # Analysis quality confidence
        if len(credibility_analysis['flags']) <= 2:
            confidence += 0.2
        
        if len(diversity_analysis['flags']) <= 1:
            confidence += 0.15
        
        if len(coordination_analysis['flags']) <= 1:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _assess_risk_level(self, score: float, flags: List[str]) -> str:
        """Assess overall risk level"""
        high_risk_flags = [
            'suspicious_dominance', 'suspicious_coordination', 
            'artificial_amplification', 'no_trusted_sources'
        ]
        
        critical_flags = any(flag in flags for flag in high_risk_flags)
        
        if score >= 0.7 and not critical_flags:
            return "low"
        elif score >= 0.4 and not critical_flags:
            return "medium"
        elif score >= 0.2:
            return "high"
        else:
            return "very_high"
    
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