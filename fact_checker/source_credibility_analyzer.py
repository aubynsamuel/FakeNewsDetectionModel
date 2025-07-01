import re
from urllib.parse import urlparse
from utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS


class SourceCredibilityAnalyzer:
    """Simplified source credibility analyzer - returns only the score"""
    
    def __init__(self):
        # Weighted scoring system
        self.weights = {
            'tld_credibility': 0.4,
            'domain_structure': 0.3,
            'news_indicators': 0.2,
            'domain_age_indicators': 0.15,
            'subdomain_analysis': 0.1
        }

        # Suspicious patterns
        self._suspicious_patterns = [
            (re.compile(r'\d{4,}'), 0.8),
            (re.compile(r'(fake|hoax|scam|click|bait|spam|phishing)', re.IGNORECASE), 0.9),
            (re.compile(r'[a-z]+\d+[a-z]+\d+', re.IGNORECASE), 0.7),
            (re.compile(r'(xxx|porn|adult|sex)', re.IGNORECASE), 0.6),
            (re.compile(r'(free|download|crack|hack)', re.IGNORECASE), 0.5),
            (re.compile(r'[0-9]{1,3}-[0-9]{1,3}-[0-9]{1,3}', re.IGNORECASE), 0.8),
            (re.compile(r'(temp|tmp|test|demo)', re.IGNORECASE), 0.4)
        ]

        # TLD scores
        self.tld_scores = {
            '.edu': 0.9, '.gov': 0.95, '.mil': 0.9,
            '.org': 0.7, '.ac.uk': 0.8, '.edu.au': 0.8,
            '.com': 0.3, '.net': 0.25, '.co.uk': 0.4, '.com.au': 0.4,
            '.de': 0.4, '.fr': 0.4, '.ca': 0.4, '.jp': 0.4,
            '.info': 0.1, '.biz': 0.1, '.name': 0.05,
            '.tk': -0.6, '.ml': -0.6, '.ga': -0.6, '.cf': -0.6,
            '.pw': -0.4, '.top': -0.3, '.click': -0.5, '.download': -0.4,
            '.stream': -0.3, '.review': -0.2, '.date': -0.3, '.racing': -0.4
        }

        # News indicators
        self.news_indicators = {
            'news': 0.3, 'times': 0.3, 'post': 0.25, 'herald': 0.2,
            'gazette': 0.2, 'journal': 0.2, 'tribune': 0.2, 'chronicle': 0.2,
            'report': 0.15, 'press': 0.2, 'media': 0.1, 'broadcast': 0.15,
            'reuters': 0.4, 'associated': 0.3, 'wire': 0.2
        }

    def analyze_domain_credibility(self, domain: str) -> float:
        """Get credibility score for domain"""
        domain = domain.lower().strip()
        
        # Handle URLs by extracting domain
        if domain.startswith(('http://', 'https://')):
            parsed = urlparse(domain)
            domain = parsed.netloc.lower()
        
        # Remove www prefix
        if domain.startswith('www.'):
            domain = domain[4:]

        # Check trusted domains
        if domain in TRUSTED_DOMAINS:
            return 0.95

        # Check suspicious domains
        if domain in SUSPICIOUS_DOMAINS:
            return 0.05

        # Calculate score components
        tld_score = self._get_tld_score(domain)
        structure_score = self._get_structure_score(domain)
        news_score = self._get_news_score(domain)
        establishment_score = self._get_establishment_score(domain)
        subdomain_score = self._get_subdomain_score(domain)
        
        # Start with base score and apply weighted components
        base_score = 0.2
        final_score = base_score
        final_score += tld_score * self.weights['tld_credibility']
        final_score += structure_score * self.weights['domain_structure'] 
        final_score += news_score * self.weights['news_indicators']
        final_score += establishment_score * self.weights['domain_age_indicators']
        final_score += subdomain_score * self.weights['subdomain_analysis']
        
        return max(0.0, min(1.0, round(final_score, 2)))

    def _get_tld_score(self, domain: str) -> float:
        """Get TLD score"""
        for tld, score in self.tld_scores.items():
            if domain.endswith(tld):
                return score
        return -0.1  # Unknown TLD

    def _get_structure_score(self, domain: str) -> float:
        """Get domain structure score"""
        suspicious_score = 0
        
        for pattern, severity in self._suspicious_patterns:
            if pattern.search(domain):
                suspicious_score -= severity * 0.3
        
        # Additional checks
        if len(domain.split('.')[0]) < 3:
            suspicious_score -= 0.2
            
        if domain.count('-') > 2:
            suspicious_score -= 0.15
            
        return max(-0.8, suspicious_score)

    def _get_news_score(self, domain: str) -> float:
        """Get news indicators score"""
        score = 0
        for indicator, weight in self.news_indicators.items():
            if indicator in domain:
                score += weight
        return min(0.4, score)

    def _get_establishment_score(self, domain: str) -> float:
        """Get establishment indicators score"""
        score = 0
        
        if any(word in domain for word in ['university', 'college', 'institute', 'foundation']):
            score += 0.3
            
        if any(word in domain for word in ['library', 'museum', 'archive']):
            score += 0.2
            
        if any(word in domain for word in ['research', 'study', 'science']):
            score += 0.15
            
        return min(0.3, score)

    def _get_subdomain_score(self, domain: str) -> float:
        """Get subdomain score"""
        parts = domain.split('.')
        
        if len(parts) <= 2:
            return 0.1
        elif len(parts) > 4:
            return -0.15
        else:
            return 0

if __name__ == "__main__":
    analyzer = SourceCredibilityAnalyzer()
    # domains_to_analyze = ["ghanaweb.com"]
    domain = input("Enter a domain to check credibility: ")
    # for domain in domains_to_analyze:
    result = analyzer.analyze_domain_credibility(domain)
    print(f"{domain} -> {result:.2f}")
