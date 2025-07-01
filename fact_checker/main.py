import time
import random
from datetime import datetime
from typing import Dict
import numpy as np
from googlesearch import search

from clickbait_checker import AdvancedClickbaitDetector
from source_credibility_analyzer import SourceCredibilityAnalyzer
from claim_verifier import ClaimVerifier
from network_analyzer import NetworkAnalyzer
from utils import extract_domain, remove_source_artifacts_fast

class EnhancedFactChecker:
    """Main enhanced fact checker with ML integration"""

    def __init__(self):
        self.clickbait_detector = AdvancedClickbaitDetector()
        self.source_analyzer = SourceCredibilityAnalyzer()
        self.claim_verifier = ClaimVerifier()  # Renamed and repurposed
        self.network_analyzer = NetworkAnalyzer()

        # print("\U0001F680 Enhanced ML-Powered Fact Checker Initialized")

    def comprehensive_verify(self, raw_headline: str, results_to_check: int = 10) -> Dict:
        """Comprehensive fact-checking with ML integration"""
        print(f"\n\U0001F50D Comprehensive Analysis: \"{raw_headline}\"")
        print("=" * 80)
        # headline = remove_source_artifacts_fast(raw_headline)

        analysis_results = {
            "headline": raw_headline,
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "final_verdict": {},
        }

        # 1. Clickbait detection
        print("\U0001F916 ML Clickbait Analysis...")
        clickbait_score = self.clickbait_detector.detect_clickbait_score(raw_headline)
        print(f"   Clickbait Score: {clickbait_score:.3f}")

        # 3. Web search
        print("\U0001F50E Searching and analyzing sources...")
        time.sleep(random.uniform(1.5, 3.0))

        try:
            search_results = list(search(raw_headline, num_results=results_to_check, lang="en"))
            print(f"   Found {len(search_results)} search results")
        except Exception as e:
            print(f"   ❌ Search error: {e}")
            search_results = []

        if not search_results:
            print("⚠️ No search results found. Assigning low credibility by default.")
            return {
                "headline": raw_headline,
                "timestamp": datetime.now().isoformat(),
                "final_verdict": {
                    "verdict": "🚫 HIGHLY QUESTIONABLE",
                    "confidence": "Very High",
                    "score": 0.1,
                    "components": {}
                }
            }

        # 4. Source credibility
        print("\U0001F4CA Analyzing source credibility...")
        source_scores = []
        trusted_count = 0
        suspicious_count = 0

        for i, url in enumerate(search_results[:results_to_check]):
            domain = extract_domain(url)
            credibility = self.source_analyzer.analyze_domain_credibility(domain)
            source_scores.append(credibility["score"])

            if credibility["score"] > 0.7:
                trusted_count += 1
                print(f"   {i+1}. {domain} ✅ ({credibility['score']:.2f})")
            elif credibility["score"] < 0.3:
                suspicious_count += 1
                print(f"   {i+1}. {domain} ❌ ({credibility['score']:.2f})")
            else:
                print(f"   {i+1}. {domain} ❓ ({credibility['score']:.2f})")

        avg_source_credibility = np.mean(source_scores) if source_scores else 0.1

        # 5. Network analysis
        print("\U0001F310 Network Propagation Analysis...")
        network_analysis = self.network_analyzer.analyze_propagation_pattern(raw_headline, search_results)
        print(f"   Propagation Score: {network_analysis['score']:.3f}")
        print(f"   Domain Diversity: {network_analysis['domain_diversity']:.3f}")

        # 6. Claim verification (no extractor used)
        print("✅ Verifying Claims...")
        verification = self.claim_verifier.verify_claim_against_sources(raw_headline, search_results)
        claim_verification_score = verification.get("score", 0.1)
        print(f"   '{raw_headline}': {claim_verification_score:.3f}")

        # 7. Weighted scoring
        components = {
            "claim_verification": (claim_verification_score, 0.40),
            "source_credibility": (avg_source_credibility, 0.25),
            "clickbait_detection": (1 - clickbait_score, 0.25),
            "network_propagation": (network_analysis["score"], 0.10),
        }

        final_score = sum(score * weight for score, weight in components.values())

        # 8. Verdict
        if final_score >= 0.75:
            verdict = "✅ Credible — Backed by Evidence"
            confidence = "Very High"
        elif final_score >= 0.60:
            verdict = "🟢 Likely True — Supported by Sources"
            confidence = "High"
        elif final_score >= 0.45:
            verdict = "⚠️ Unclear — Conflicting Information"
            confidence = "Moderate"
        elif final_score >= 0.30:
            verdict = "🟠 Doubtful — Weak or Biased Evidence"
            confidence = "Low"
        else:
            verdict = "🚫 False or Misleading — No Basis Found"
            confidence = "Very Low"

        analysis_results["final_verdict"] = {
            "verdict": verdict,
            "confidence": confidence,
            "score": final_score,
            "components": components
        }

        analysis_results["components"] = {
            "clickbait": {"score": clickbait_score, "weight": 0.25},
            "source_credibility": {
                "score": avg_source_credibility,
                "trusted_count": trusted_count,
                "suspicious_count": suspicious_count,
                "weight": 0.25
            },
            "network": network_analysis,
            "claim_verification": {
                "score": claim_verification_score,
                "verified_claims": 1,
                "weight": 0.40
            }
        }

        # 9. Summary
        print(f"\n📈 COMPREHENSIVE ANALYSIS RESULTS:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🎯 Final Score: {final_score:.3f}/1.000")
        print(f"🏆 Verdict: {verdict}")
        print(f"📊 Confidence: {confidence}")

        print(f"\n🔍 Component Breakdown:")
        for component, (score, weight) in components.items():
            print(f"   • {component.replace('_', ' ').title()}: {score:.3f} (weight: {weight:.0%})")

        print(f"\n📋 Summary:")
        print(f"   • Trusted Sources: {trusted_count}")
        print(f"   • Suspicious Sources: {suspicious_count}")
        print(f"   • Claims Verified: 1")
        print(f"   • Clickbait Score: {clickbait_score:.3f}")
        print(f"   • Domain Diversity: {network_analysis['domain_diversity']:.3f}")

        return analysis_results


# Example usage and testing
if __name__ == "__main__":
    checker = EnhancedFactChecker()
    
    print("🚀 AI-Powered Fake News Detector")
    print("This system combines multiple ML techniques with fact-checking")
    print()
    
    # Test examples
    test_headlines = [
        "SHOCKING: Scientists Don't Want You to Know This One Weird Trick!",
        "Parliament approves new healthcare funding for rural communities",
        "BREAKING: Leaked footage shows aliens landing in Ghana!",
        "Bank of Ghana announces new monetary policy measures"
    ]
    
    choice = input("Test with sample headlines? (y/n): ").lower()
    
    if choice == 'y':
        for headline in test_headlines:
            result = checker.comprehensive_verify(headline, results_to_check=5)
            print("\n" + "="*80 + "\n")
            time.sleep(2)  # Pause between tests
    else:
        while True:
            user_input = input("Enter news headline to verify (or 'quit' to exit): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input:
                result = checker.comprehensive_verify(user_input)
                print("\n" + "="*80 + "\n")