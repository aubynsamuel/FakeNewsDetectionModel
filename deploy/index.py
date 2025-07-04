import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from googlesearch import search

from deploy.main.claim_verifier import ClaimVerifier
from deploy.main.network_analyzer import NetworkAnalyzer
from deploy.main.source_credibility_analyzer import SourceCredibilityAnalyzer
from deploy.utils.general_utils import extract_domain
from deploy.main.predict_clickbait import predict_clickbait


class FakeNewsDetector:
    """Main enhanced fact checker with ML integration"""

    def __init__(self):
        try:
            self.source_analyzer = SourceCredibilityAnalyzer()
            self.claim_verifier = ClaimVerifier()
            self.network_analyzer = NetworkAnalyzer()
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert any numeric value to Python float"""
        try:
            if isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return default
        except (ValueError, TypeError):
            return default

    def _analyze_clickbait(self, headline: str) -> float:
        """Analyzes the headline for clickbait characteristics."""
        print("🧠 ML Clickbait Analysis...")
        try:
            _, clickbait_score, _ = predict_clickbait(headline)
            clickbait_score = self._to_float(clickbait_score, 0.5)
            print(f"   Clickbait Score: {clickbait_score:.2f}")
            return clickbait_score
        except Exception as e:
            print(f"   ❌ Clickbait analysis error: {e}")
            return 0.5  # Default moderate score

    def _search_for_sources(self, headline: str, num_results: int) -> List[str]:
        """Searches the web for sources related to the headline."""
        print("🔎 Searching and analyzing sources...")
        try:
            time.sleep(random.uniform(1.5, 3.0))
            search_results = list(search(headline, num_results=num_results, lang="en"))
            print(f"   Found {len(search_results)} search results")
            return search_results
        except Exception as e:
            print(f"   ❌ Search error: {e}")
            return []

    def _analyze_source_credibility(
        self, search_results: List[str]
    ) -> Tuple[float, int, int]:
        """Analyzes the credibility of the found source domains."""
        print("📊 Analyzing source credibility...")

        if not search_results:
            print("   ❌ No search results to analyze")
            return 0.1, 0, 0

        source_scores = []
        trusted_count = 0
        suspicious_count = 0

        for i, url in enumerate(search_results):
            try:
                domain = extract_domain(url)
                credibility_score = self.source_analyzer.analyze_domain_credibility(
                    domain
                )
                credibility_score = self._to_float(credibility_score, 0.5)
                source_scores.append(credibility_score)

                if credibility_score > 0.7:
                    trusted_count += 1
                    print(f"   {i+1}. {domain} ✅ ({credibility_score:.2f})")
                elif credibility_score < 0.3:
                    suspicious_count += 1
                    print(f"   {i+1}. {domain} ❌ ({credibility_score:.2f})")
                else:
                    print(f"   {i+1}. {domain} ❓ ({credibility_score:.2f})")
            except Exception as e:
                print(f"   ❌ Error analyzing {url}: {e}")
                source_scores.append(0.3)  # Default neutral score

        # Use regular Python mean instead of np.mean
        avg_credibility = (
            sum(source_scores) / len(source_scores) if source_scores else 0.1
        )
        return avg_credibility, trusted_count, suspicious_count

    def _analyze_network_propagation(
        self, search_results: List[str]
    ) -> Dict[str, float]:
        """Analyzes the propagation pattern of the news across the network."""
        print("🌐 Network Propagation Analysis...")

        if not search_results:
            print("   ❌ No search results for network analysis")
            return {"score": 0.1, "domain_diversity": 0.0}

        try:
            network_analysis = self.network_analyzer.analyze_propagation_pattern(
                search_results
            )

            # Convert all values to Python floats
            result = {
                "score": self._to_float(network_analysis.get("score", 0.1)),
                "domain_diversity": self._to_float(
                    network_analysis.get("domain_diversity", 0.0)
                ),
            }

            print(f"   Propagation Score: {result['score']:.2f}")
            print(f"   Domain Diversity: {result['domain_diversity']:.2f}")
            return result
        except Exception as e:
            print(f"   ❌ Network analysis error: {e}")
            return {"score": 0.1, "domain_diversity": 0.0}

    def _verify_claim(self, headline: str, search_results: List[str]) -> float:
        """Verifies the claim against the content of the found sources."""
        print("✅ Verifying Claims...")

        if not search_results:
            print("   ❌ No search results for claim verification")
            return 0.1

        try:
            verification = self.claim_verifier.verify_claim_against_sources(
                headline, search_results
            )
            claim_verification_score = self._to_float(verification.get("score", 0.1))
            print(f"   '{headline}': {claim_verification_score:.2f}")
            return claim_verification_score
        except Exception as e:
            print(f"   ❌ Claim verification error: {e}")
            return 0.1

    def _calculate_final_score_and_verdict(
        self, component_scores: Dict[str, float]
    ) -> Tuple[float, str, str]:
        """Calculates the final weighted score and determines the verdict."""
        weights = {
            "claim_verification": 0.35,
            "source_credibility": 0.25,
            "clickbait_detection": 0.25,
            "network_propagation": 0.15,
        }

        final_score = sum(
            component_scores.get(component, 0.0) * weight
            for component, weight in weights.items()
        )

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

        return final_score, verdict, confidence

    def _print_summary(self, results: Dict):
        """Prints a formatted summary of the analysis results."""
        final_verdict = results["final_verdict"]
        components = results["components"]

        print(f"📈 COMPREHENSIVE ANALYSIS RESULTS:")
        print(
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        print(f"🎯 Final Score: {final_verdict['score']:.2f}/1.000")
        print(f"🏆 Verdict: {final_verdict['verdict']}")
        print(f"📊 Confidence: {final_verdict['confidence']}")

        print(f"🔍 Component Breakdown:")
        for component, score in final_verdict["components"].items():
            print(f"   • {component.replace('_', ' ').title()}: {score:.2f}")

        print(f"📋 Summary:")
        print(
            f"   • Trusted Sources: {components['source_credibility']['trusted_count']}"
        )
        print(
            f"   • Suspicious Sources: {components['source_credibility']['suspicious_count']}"
        )
        print(
            f"   • Clickbait Score: {components['clickbait']['score']:.2f} (lower is better)"
        )
        print(f"   • Domain Diversity: {components['network']['domain_diversity']:.2f}")

    def comprehensive_verify(
        self, raw_headline: str, results_to_check: int = 8
    ) -> Dict:
        """
        Comprehensive fact-checking with ML integration.
        This method orchestrates the analysis by calling various specialized components.
        """
        print(f'\n🔎 Comprehensive Analysis: "{raw_headline}"')
        print("=" * 80)

        if not raw_headline or not raw_headline.strip():
            print("❌ Empty or invalid headline provided")
            return {
                "headline": "",
                "timestamp": datetime.now().isoformat(),
                "final_verdict": {
                    "verdict": "❌ Invalid Input",
                    "confidence": "Very High",
                    "score": 0.0,
                    "components": {
                        "claim_verification": 0.0,
                        "source_credibility": 0.0,
                        "clickbait_detection": 0.0,
                        "network_propagation": 0.0,
                    },
                },
                "components": {
                    "clickbait": {"score": 0.0},
                    "source_credibility": {
                        "score": 0.0,
                        "trusted_count": 0,
                        "suspicious_count": 0,
                    },
                    "network": {"score": 0.0, "domain_diversity": 0.0},
                    "claim_verification": {"score": 0.0},
                },
            }

        # Step 1: Search for sources
        search_results = self._search_for_sources(raw_headline, results_to_check)

        if not search_results:
            print("⚠️ No search results found. Assigning low credibility by default.")
            return {
                "headline": raw_headline,
                "timestamp": datetime.now().isoformat(),
                "final_verdict": {
                    "verdict": "🚫 HIGHLY QUESTIONABLE",
                    "confidence": "Very High",
                    "score": 0.1,
                    "components": {
                        "claim_verification": 0.1,
                        "source_credibility": 0.1,
                        "clickbait_detection": 0.1,
                        "network_propagation": 0.1,
                    },
                },
                "components": {
                    "clickbait": {"score": 0.5},
                    "source_credibility": {
                        "score": 0.1,
                        "trusted_count": 0,
                        "suspicious_count": 0,
                    },
                    "network": {"score": 0.1, "domain_diversity": 0.0},
                    "claim_verification": {"score": 0.1},
                },
            }

        # Step 2: Run all analysis components
        clickbait_score = self._analyze_clickbait(raw_headline)
        avg_source_credibility, trusted_count, suspicious_count = (
            self._analyze_source_credibility(search_results)
        )
        network_analysis = self._analyze_network_propagation(search_results)
        claim_verification_score = self._verify_claim(raw_headline, search_results)

        # Step 3: Consolidate component scores (ensure all are Python floats)
        component_scores = {
            "claim_verification": claim_verification_score,
            "source_credibility": avg_source_credibility,
            "clickbait_detection": 1.0 - clickbait_score,  # Invert score
            "network_propagation": network_analysis["score"],
        }

        # Step 4: Calculate final score and verdict
        final_score, verdict, confidence = self._calculate_final_score_and_verdict(
            component_scores
        )

        # Step 5: Build the exact JSON structure you specified
        analysis_results = {
            "headline": raw_headline,
            "timestamp": datetime.now().isoformat(),
            "final_verdict": {
                "verdict": verdict,
                "confidence": confidence,
                "score": round(final_score, 2),
                "components": {
                    "claim_verification": round(
                        component_scores["claim_verification"], 2
                    ),
                    "source_credibility": round(
                        component_scores["source_credibility"], 2
                    ),
                    "clickbait_detection": round(
                        component_scores["clickbait_detection"], 2
                    ),
                    "network_propagation": round(
                        component_scores["network_propagation"], 2
                    ),
                },
            },
            "components": {
                "clickbait": {"score": round(clickbait_score, 2)},
                "source_credibility": {
                    "score": round(avg_source_credibility, 2),
                    "trusted_count": trusted_count,
                    "suspicious_count": suspicious_count,
                },
                "network": {
                    "score": round(network_analysis["score"], 2),
                    "domain_diversity": round(network_analysis["domain_diversity"], 2),
                },
                "claim_verification": {"score": round(claim_verification_score, 2)},
            },
        }

        # self._print_summary(analysis_results)
        return analysis_results


# Example usage and testing
if __name__ == "__main__":
    try:
        checker = FakeNewsDetector()

        print("🚀 AI-Powered Fake News Detector")
        print("This system combines multiple ML techniques with fact-checking")
        print()

        # Test examples
        test_headlines = [
            "SHOCKING: Scientists Don't Want You to Know This One Weird Trick!",
            "Parliament approves new healthcare funding for rural communities",
            "BREAKING: Leaked footage shows aliens landing in Ghana!",
            "Bank of Ghana announces new monetary policy measures",
        ]

        choice = input("Test with sample headlines? (y/n): ").lower()

        if choice == "y":
            for headline in test_headlines:
                try:
                    result = checker.comprehensive_verify(headline, results_to_check=4)
                    print(f"\n📊 Result for: {headline}")
                    print(f"Score: {result['final_verdict']['score']}")
                    print(f"Verdict: {result['final_verdict']['verdict']}")
                    print("\n" + "=" * 80 + "\n")
                    time.sleep(5)  # Pause between tests
                except Exception as e:
                    print(f"❌ Error processing headline '{headline}': {e}")
        else:
            while True:
                try:
                    user_input = input(
                        "Enter news headline to verify (or 'quit' to exit): "
                    ).strip()
                    if user_input.lower() in ["quit", "exit", "q"]:
                        break
                    if user_input:
                        result = checker.comprehensive_verify(user_input)
                        print(f"\n📊 Analysis Result:")
                        print(f"Score: {result['final_verdict']['score']}")
                        print(f"Verdict: {result['final_verdict']['verdict']}")
                        print(f"{result['final_verdict']['components']}\n")
                        print(result)
                        print("\n" + "=" * 80 + "\n")
                    else:
                        print("Please enter a valid headline.")
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
    except Exception as e:
        print(f"❌ Failed to initialize fact checker: {e}")
        print("Please ensure all required modules are available.")
