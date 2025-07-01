import re
from datetime import timedelta
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS


class SourceCredibilityAnalyzer:
    """Enhanced source credibility analyzer with improved scoring and weighting"""
    
    def __init__(self):
        self.domain_scores = {}
        self.update_interval = timedelta(hours=24)
        self.last_update = {}

        # Weighted scoring system
        self.weights = {
            'domain_whitelist': 1.0,      # Highest priority
            'domain_blacklist': -1.0,     # Immediate disqualification
            'tld_credibility': 0.4,       # Significant weight
            'domain_structure': 0.3,      # Important for detection
            'news_indicators': 0.2,       # Moderate boost
            'domain_age_indicators': 0.15, # Additional factor
            'subdomain_analysis': 0.1     # Minor factor
        }

        # Enhanced suspicious patterns with severity levels
        self._suspicious_patterns = [
            (re.compile(r'\d{4,}'), 0.8, "Long numeric sequences"),
            (re.compile(r'(fake|hoax|scam|click|bait|spam|phishing)', re.IGNORECASE), 0.9, "Malicious keywords"),
            (re.compile(r'[a-z]+\d+[a-z]+\d+', re.IGNORECASE), 0.7, "Complex alphanumeric mixing"),
            (re.compile(r'(xxx|porn|adult|sex)', re.IGNORECASE), 0.6, "Adult content indicators"),
            (re.compile(r'(free|download|crack|hack)', re.IGNORECASE), 0.5, "Suspicious service keywords"),
            (re.compile(r'[0-9]{1,3}-[0-9]{1,3}-[0-9]{1,3}', re.IGNORECASE), 0.8, "IP-like patterns"),
            (re.compile(r'(temp|tmp|test|demo)', re.IGNORECASE), 0.4, "Temporary domain indicators")
        ]

        # Enhanced TLD classification with more granular scoring
        self.tld_scores = {
            # High trust TLDs
            '.edu': 0.9, '.gov': 0.95, '.mil': 0.9,
            # Established news/org TLDs  
            '.org': 0.7, '.ac.uk': 0.8, '.edu.au': 0.8,
            # Commercial but reliable
            '.com': 0.3, '.net': 0.25, '.co.uk': 0.4, '.com.au': 0.4,
            '.de': 0.4, '.fr': 0.4, '.ca': 0.4, '.jp': 0.4,
            # Medium trust
            '.info': 0.1, '.biz': 0.1, '.name': 0.05,
            # Low trust / suspicious
            '.tk': -0.6, '.ml': -0.6, '.ga': -0.6, '.cf': -0.6,
            '.pw': -0.4, '.top': -0.3, '.click': -0.5, '.download': -0.4,
            '.stream': -0.3, '.review': -0.2, '.date': -0.3, '.racing': -0.4
        }

        # Legitimate news indicators with weights
        self.news_indicators = {
            'news': 0.3, 'times': 0.3, 'post': 0.25, 'herald': 0.2,
            'gazette': 0.2, 'journal': 0.2, 'tribune': 0.2, 'chronicle': 0.2,
            'report': 0.15, 'press': 0.2, 'media': 0.1, 'broadcast': 0.15,
            'reuters': 0.4, 'associated': 0.3, 'wire': 0.2
        }

    def analyze_domain_credibility(self, domain: str) -> Dict:
        """Enhanced domain credibility analysis with weighted scoring"""
        domain = domain.lower().strip()
        
        # Handle URLs by extracting domain
        if domain.startswith(('http://', 'https://')):
            parsed = urlparse(domain)
            domain = parsed.netloc.lower()
        
        # Remove www prefix for consistency
        if domain.startswith('www.'):
            domain = domain[4:]

        # Check whitelisted domains first
        if domain in TRUSTED_DOMAINS:
            return {
                "score": 0.95, 
                "reason": "Verified trusted source",
                "confidence": "high",
                "risk_level": "very_low"
            }

        # Check blacklisted domains
        if domain in SUSPICIOUS_DOMAINS:
            return {
                "score": 0.05, 
                "reason": "Known suspicious/malicious source",
                "confidence": "high", 
                "risk_level": "very_high"
            }

        # Multi-factor scoring for unknown domains
        score_components = self._calculate_score_components(domain)
        final_score = self._compute_weighted_score(score_components)
        
        return {
            "score": max(0.0, min(1.0, round(final_score, 2))),
            "reason": self._generate_detailed_reason(score_components),
            "confidence": self._assess_confidence(score_components),
            "risk_level": self._assess_risk_level(final_score),
            "components": score_components
        }

    def _calculate_score_components(self, domain: str) -> Dict:
        """Calculate individual scoring components"""
        components = {}
        
        # TLD Analysis
        components['tld'] = self._analyze_tld_credibility(domain)
        
        # Domain structure analysis
        components['structure'] = self._analyze_domain_structure(domain)
        
        # News indicators
        components['news_indicators'] = self._check_news_indicators(domain)
        
        # Age/establishment indicators
        components['establishment'] = self._check_establishment_indicators(domain)
        
        # Subdomain analysis
        components['subdomain'] = self._analyze_subdomain_credibility(domain)
        
        return components

    def _compute_weighted_score(self, components: Dict) -> float:
        """Compute final weighted score"""
        # Start with a lower base score for unknown domains
        base_score = 0.2
        
        # Apply weighted components
        score = base_score
        score += components['tld']['score'] * self.weights['tld_credibility']
        score += components['structure']['score'] * self.weights['domain_structure'] 
        score += components['news_indicators']['score'] * self.weights['news_indicators']
        score += components['establishment']['score'] * self.weights['domain_age_indicators']
        score += components['subdomain']['score'] * self.weights['subdomain_analysis']
        
        return score

    def _analyze_tld_credibility(self, domain: str) -> Dict:
        """Enhanced TLD analysis"""
        for tld, score in self.tld_scores.items():
            if domain.endswith(tld):
                confidence = "high" if abs(score) > 0.5 else "medium"
                return {
                    "score": score,
                    "tld": tld,
                    "confidence": confidence,
                    "description": self._get_tld_description(score)
                }
        
        # Unknown TLD - slightly suspicious
        return {
            "score": -0.1,
            "tld": "unknown",
            "confidence": "low", 
            "description": "Unknown or uncommon TLD"
        }

    def _analyze_domain_structure(self, domain: str) -> Dict:
        """Analyze domain structure for suspicious patterns"""
        suspicious_score = 0
        detected_patterns = []
        
        for pattern, severity, description in self._suspicious_patterns:
            if pattern.search(domain):
                suspicious_score -= severity * 0.3  # Scale down impact
                detected_patterns.append(description)
        
        # Additional checks
        if len(domain.split('.')[0]) < 3:  # Very short domain names
            suspicious_score -= 0.2
            detected_patterns.append("Very short domain name")
            
        if domain.count('-') > 2:  # Too many hyphens
            suspicious_score -= 0.15
            detected_patterns.append("Excessive hyphens")
            
        return {
            "score": max(-0.8, suspicious_score),  # Cap negative impact
            "patterns": detected_patterns,
            "confidence": "high" if detected_patterns else "medium"
        }

    def _check_news_indicators(self, domain: str) -> Dict:
        """Check for legitimate news/media indicators"""
        score = 0
        found_indicators = []
        
        for indicator, weight in self.news_indicators.items():
            if indicator in domain:
                score += weight
                found_indicators.append(indicator)
        
        return {
            "score": min(0.4, score),  # Cap positive boost
            "indicators": found_indicators,
            "confidence": "medium" if found_indicators else "low"
        }

    def _check_establishment_indicators(self, domain: str) -> Dict:
        """Check for indicators of established organizations"""
        score = 0
        indicators = []
        
        # Established patterns
        if any(word in domain for word in ['university', 'college', 'institute', 'foundation']):
            score += 0.3
            indicators.append("Educational institution")
            
        if any(word in domain for word in ['library', 'museum', 'archive']):
            score += 0.2
            indicators.append("Cultural institution")
            
        if any(word in domain for word in ['research', 'study', 'science']):
            score += 0.15
            indicators.append("Research organization")
            
        return {
            "score": min(0.3, score),
            "indicators": indicators,
            "confidence": "medium" if indicators else "low"
        }

    def _analyze_subdomain_credibility(self, domain: str) -> Dict:
        """Analyze subdomain patterns"""
        parts = domain.split('.')
        score = 0
        
        if len(parts) <= 2:  # Simple domain structure
            score += 0.1
        elif len(parts) > 4:  # Too many subdomains
            score -= 0.15
            
        return {
            "score": score,
            "subdomain_count": len(parts) - 2,
            "confidence": "low"
        }

    def _get_tld_description(self, score: float) -> str:
        """Get description for TLD score"""
        if score >= 0.8:
            return "Highly trusted institutional TLD"
        elif score >= 0.5:
            return "Trusted organizational TLD"
        elif score >= 0.2:
            return "Standard commercial TLD"
        elif score >= 0:
            return "Neutral TLD"
        else:
            return "Suspicious or low-trust TLD"

    def _generate_detailed_reason(self, components: Dict) -> str:
        """Generate detailed reasoning for the score"""
        reasons = []
        
        tld_info = components['tld']
        if abs(tld_info['score']) > 0.2:
            reasons.append(f"TLD: {tld_info['description']}")
            
        structure_info = components['structure']
        if structure_info['patterns']:
            reasons.append(f"Suspicious patterns: {', '.join(structure_info['patterns'])}")
            
        news_info = components['news_indicators']
        if news_info['indicators']:
            reasons.append(f"News indicators: {', '.join(news_info['indicators'])}")
            
        est_info = components['establishment']
        if est_info['indicators']:
            reasons.append(f"Institutional: {', '.join(est_info['indicators'])}")
            
        return "; ".join(reasons) if reasons else "Unknown domain with limited indicators"

    def _assess_confidence(self, components: Dict) -> str:
        """Assess confidence level of the analysis"""
        high_confidence_count = sum(1 for comp in components.values() 
                                  if comp.get('confidence') == 'high')
        
        if high_confidence_count >= 2:
            return "high"
        elif high_confidence_count >= 1:
            return "medium"
        else:
            return "low"

    def _assess_risk_level(self, score: float) -> str:
        """Assess risk level based on final score"""
        if score >= 0.8:
            return "very_low"
        elif score >= 0.6:
            return "low" 
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "high"
        else:
            return "very_high"