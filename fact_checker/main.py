import time
import random
from datetime import datetime
from typing import Dict
import numpy as np
from googlesearch import search

from clickbait_checker import MLClickbaitDetector
from source_credibility_analyzer import SourceCredibilityAnalyzer
from claim_extractor import ClaimExtractor
from network_analyzer import NetworkAnalyzer
from temporal_analyzer import TemporalAnalyzer
from utils import extract_domain

class EnhancedFactChecker:
    """Main enhanced fact checker with ML integration"""
    
    def __init__(self):
        self.clickbait_detector = MLClickbaitDetector()
        self.source_analyzer = SourceCredibilityAnalyzer()
        self.claim_extractor = ClaimExtractor()
        self.network_analyzer = NetworkAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        
        print("🚀 Enhanced ML-Powered Fact Checker Initialized")
    
    def comprehensive_verify(self, headline: str, results_to_check: int = 10) -> Dict:
        """Comprehensive fact-checking with ML integration"""
        print(f"\n🔍 Comprehensive Analysis: \"{headline}\"")
        print("=" * 80)
        
        # Initialize results
        analysis_results = {
            "headline": headline,
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "final_verdict": {},
        }
        
        # 1. ML-based clickbait detection
        print("🤖 ML Clickbait Analysis...")
        clickbait_score = self.clickbait_detector.detect_clickbait_score(headline)
        analysis_results["components"]["clickbait"] = {
            "score": clickbait_score,
            "weight": 0.15
        }
        print(f"   Clickbait Score: {clickbait_score:.3f}")
        
        # 2. Claim extraction
        print("🎯 Extracting Claims...")
        claims = self.claim_extractor.extract_claims(headline)
        analysis_results["components"]["claims"] = claims
        print(f"   Extracted {len(claims)} verifiable claims")
        
        # 3. Temporal analysis
        print("⏰ Temporal Analysis...")
        temporal_analysis = self.temporal_analyzer.check_recycled_content(headline)
        analysis_results["components"]["temporal"] = temporal_analysis
        print(f"   Recycling Score: {temporal_analysis['score']:.3f}")
        
        # 4. Search and source analysis
        print("🔎 Searching and analyzing sources...")
        time.sleep(random.uniform(1.5, 3.0))  # Rate limiting
        
        try:
            search_results = list(search(headline, num_results=results_to_check, lang="en"))
            print(f"   Found {len(search_results)} search results")
        except Exception as e:
            print(f"   ❌ Search error: {e}")
            search_results = []
        
        # 5. Source credibility analysis
        source_scores = []
        trusted_count = 0
        suspicious_count = 0
        
        print("📊 Analyzing source credibility...")
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
        
        avg_source_credibility = np.mean(source_scores) if source_scores else 0.5
        
        # 6. Network propagation analysis
        print("🌐 Network Propagation Analysis...")
        network_analysis = self.network_analyzer.analyze_propagation_pattern(headline, search_results)
        analysis_results["components"]["network"] = network_analysis
        print(f"   Propagation Score: {network_analysis['score']:.3f}")
        print(f"   Domain Diversity: {network_analysis['domain_diversity']:.3f}")
        
        # 7. Claim verification
        print("✅ Verifying Claims...")
        claim_verification_scores = []
        for claim in claims[:3]:  # Verify top 3 claims
            verification = self.claim_extractor.verify_claim_against_sources(
                claim["text"], search_results
            )
            claim_verification_scores.append(verification["score"])
            print(f"   '{claim['text']}': {verification['score']:.3f}")
        
        avg_claim_verification = np.mean(claim_verification_scores) if claim_verification_scores else 0.5
        
        # 8. Calculate final score with enhanced weighting
        components = {
            "source_credibility": (avg_source_credibility, 0.30),
            "claim_verification": (avg_claim_verification, 0.25),
            "network_propagation": (network_analysis["score"], 0.20),
            "clickbait_detection": (1 - clickbait_score, 0.15),  # Invert: lower clickbait = higher credibility
            "temporal_consistency": (1 - temporal_analysis["score"], 0.10)  # Invert: less recycling = higher credibility
        }
        
        final_score = sum(score * weight for score, weight in components.values())
        
        # Store component analysis
        analysis_results["components"]["source_credibility"] = {
            "score": avg_source_credibility,
            "trusted_count": trusted_count,
            "suspicious_count": suspicious_count,
            "weight": 0.30
        }
        
        analysis_results["components"]["claim_verification"] = {
            "score": avg_claim_verification,
            "verified_claims": len(claim_verification_scores),
            "weight": 0.25
        }
        
        # 9. Generate verdict
        if final_score >= 0.75:
            verdict = "✅ HIGHLY CREDIBLE"
            confidence = "Very High"
        elif final_score >= 0.60:
            verdict = "✅ LIKELY CREDIBLE"
            confidence = "High"
        elif final_score >= 0.45:
            verdict = "⚠️ MIXED SIGNALS"
            confidence = "Medium"
        elif final_score >= 0.30:
            verdict = "❗ LIKELY QUESTIONABLE"
            confidence = "High"
        else:
            verdict = "🚫 HIGHLY QUESTIONABLE"
            confidence = "Very High"
        
        analysis_results["final_verdict"] = {
            "verdict": verdict,
            "confidence": confidence,
            "score": final_score,
            "components": components
        }
        
        # 10. Display results
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
        print(f"   • Claims Verified: {len(claim_verification_scores)}")
        print(f"   • Clickbait Score: {clickbait_score:.3f}")
        print(f"   • Domain Diversity: {network_analysis['domain_diversity']:.3f}")
        
        return analysis_results

# Example usage and testing
if __name__ == "__main__":
    checker = EnhancedFactChecker()
    
    print("🚀 Enhanced ML-Powered Fake News Detector")
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