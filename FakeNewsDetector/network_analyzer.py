import math
from collections import Counter
from typing import List, Dict
from .utils import TRUSTED_DOMAINS, SUSPICIOUS_DOMAINS, extract_domain

SOCIAL_AGGREGATOR_DOMAINS = {
    "facebook.com",
    "twitter.com",
    "reddit.com",
    "youtube.com",
    "instagram.com",
    "tiktok.com",
    "google.com",
    "yahoo.com",
    "msn.com",
    "aol.com",
    "linkedin.com",
    "pinterest.com",
    "snapchat.com",
    "discord.com",
    "telegram.org",
}

CONTENT_FARM_DOMAINS = {
    "buzzfeed.com",
    "clickhole.com",
    "upworthy.com",
    "viralthread.com",
    "shareably.net",
    "littlethings.com",
    "providr.com",
    "shared.com",
}


class NetworkAnalyzer:
    """Propagation pattern analyzer - returns only score, and domain_diversity"""

    def __init__(self):
        # Scoring weights
        self.weights = {"domain_credibility": 0.60, "diversity_quality": 0.40}
        self.min_sources_threshold = 3
        self.min_unique_domains = 2

    def _calculate_domain_credibility_score(self, domains: List[str]) -> float:
        """Calculate domain credibility score"""
        if not domains:
            return 0.0

        domain_counts = Counter(domains)
        total_sources = len(domains)

        # Categorize domains
        trusted_count = sum(
            count
            for domain, count in domain_counts.items()
            if domain in TRUSTED_DOMAINS
        )
        suspicious_count = sum(
            count
            for domain, count in domain_counts.items()
            if domain in SUSPICIOUS_DOMAINS
        )
        social_count = sum(
            count
            for domain, count in domain_counts.items()
            if domain in SOCIAL_AGGREGATOR_DOMAINS
        )
        content_farm_count = sum(
            count
            for domain, count in domain_counts.items()
            if domain in CONTENT_FARM_DOMAINS
        )

        # Calculate ratios
        trusted_ratio = trusted_count / total_sources
        suspicious_ratio = suspicious_count / total_sources
        social_ratio = social_count / total_sources
        content_farm_ratio = content_farm_count / total_sources
        unknown_ratio = 1 - (
            trusted_ratio + suspicious_ratio + social_ratio + content_farm_ratio
        )

        # Calculate score
        base_score = 0.15
        score = base_score
        score += trusted_ratio * 0.6
        score -= suspicious_ratio * 0.8
        score -= content_farm_ratio * 0.4
        score += social_ratio * 0.1
        score -= unknown_ratio * 0.2

        # Additional penalties
        if suspicious_ratio > 0.5:
            score -= 0.3
        if trusted_count == 0 and total_sources > 5:
            score -= 0.2
        if content_farm_ratio > 0.4:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _calculate_diversity_quality(self, domains: List[str]) -> Dict:
        """Calculate diversity quality - returns score and entropy
        Entropy here is a statistical measure of domain diversity,
        helping to assess whether a claim’s spread is broad and
        organic or narrow and potentially suspicious.
        """
        if len(domains) < 2:
            return {"score": 0.0, "entropy": 0.0}

        domain_counts = Counter(domains)
        unique_domains = len(set(domains))
        total_sources = len(domains)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in domain_counts.values():
            p = count / total_sources
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize entropy
        max_entropy = math.log2(unique_domains) if unique_domains > 1 else 0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Base diversity score
        diversity_score = normalized_entropy

        # Detect artificial patterns
        max_domain_share = max(domain_counts.values()) / total_sources

        # Single domain dominance penalty
        if max_domain_share > 0.7 and unique_domains > 3:
            diversity_score -= 0.4

        # Artificial diversity penalty
        single_mention_domains = sum(
            1 for count in domain_counts.values() if count == 1
        )
        if single_mention_domains > total_sources * 0.8 and total_sources > 10:
            diversity_score -= 0.3

        if 0.3 <= normalized_entropy <= 0.8 and unique_domains >= 3:
            diversity_score += 0.2

        return {
            "score": max(0.0, min(1.0, diversity_score)),
            "entropy": normalized_entropy,
        }

    def analyze_propagation_pattern(self, search_results: List[str]) -> Dict:
        """Analyze propagation pattern - returns score, and domain_diversity"""
        domains = []
        valid_urls = 0

        for url in search_results:
            domain = extract_domain(url)
            if domain and domain not in ["", "localhost"]:
                domains.append(domain)
                valid_urls += 1

        # Early return for insufficient data
        if len(domains) < self.min_sources_threshold:
            return {"score": 0.1, "domain_diversity": 0.0}

        # Perform analysis
        credibility_score = self._calculate_domain_credibility_score(domains)
        diversity_analysis = self._calculate_diversity_quality(domains)

        # Calculate weighted final score
        final_score = (
            credibility_score * self.weights["domain_credibility"]
            + diversity_analysis["score"] * self.weights["diversity_quality"]
        )

        # Additional quality adjustments
        unique_domains = len(set(domains))
        trusted_count = sum(1 for d in domains if d in TRUSTED_DOMAINS)
        suspicious_count = sum(1 for d in domains if d in SUSPICIOUS_DOMAINS)

        if trusted_count >= 3 and suspicious_count == 0:
            final_score += 0.1
        elif suspicious_count > trusted_count:
            final_score -= 0.15

        if unique_domains < self.min_unique_domains:
            final_score = min(final_score, 0.3)

        final_score = max(0.0, min(1.0, final_score))

        return {
            "score": round(final_score, 3),
            "domain_diversity": round(diversity_analysis["entropy"], 3),
        }


if __name__ == "__main__":
    analyzer = NetworkAnalyzer()

    # Test Case 1: Mixed credible and suspicious domains
    search_results_1 = [
        "https://reuters.com/news/article1",
        "https://bbc.com/news/article2",
        "https://ghanaweb.com/article3",
        "https://cnn.com/article4",
        "https://naturalnews.com/fake1",
        "https://infowars.com/fake2",
    ]
    print("\nTest Case 1: Mixed credible and suspicious")
    result1 = analyzer.analyze_propagation_pattern(search_results_1)
    print(f"Result: {result1}")

    # Test Case 2: Mostly trusted domains
    search_results_2 = [
        "https://bbc.com/article",
        "https://cnn.com/article",
        "https://reuters.com/article",
        "https://nytimes.com/article",
        "https://ghanaweb.com/article",
    ]
    print("\nTest Case 2: Mostly trusted domains")
    result2 = analyzer.analyze_propagation_pattern(search_results_2)
    print(f"Result: {result2}")

    # Test Case 3: Mostly suspicious and content farms
    search_results_3 = [
        "https://infowars.com/fake",
        "https://naturalnews.com/fake",
        "https://clickhole.com/funny",
        "https://upworthy.com/clickbait",
        "https://shared.com/share",
    ]
    print("\nTest Case 3: Suspicious and content farm heavy")
    result3 = analyzer.analyze_propagation_pattern(search_results_3)
    print(f"Result: {result3}")

    # Test Case 4: Low diversity (same domain repeated)
    search_results_4 = [
        "https://buzzfeed.com/post1",
        "https://buzzfeed.com/post2",
        "https://buzzfeed.com/post3",
        "https://buzzfeed.com/post4",
        "https://buzzfeed.com/post5",
    ]
    print("\nTest Case 4: Low domain diversity")
    result4 = analyzer.analyze_propagation_pattern(search_results_4)
    print(f"Result: {result4}")

    # Test Case 5: Not enough sources
    search_results_5 = ["https://cnn.com/article1"]
    print("\nTest Case 5: Insufficient results")
    result5 = analyzer.analyze_propagation_pattern(search_results_5)
    print(f"Result: {result5}")
