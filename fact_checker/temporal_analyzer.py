import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from collections import defaultdict
from functools import lru_cache

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

# Define stopwords for headline cleaning (you might want to expand this)
STOPWORDS = set([
    "a", "an", "the", "and", "but", "or", "of", "in", "on", "for", "with",
    "at", "by", "from", "up", "about", "as", "into", "to", "is", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "can", "could", "should", "may", "might",
    "must", "etc", "new", "old", "recent", "breaking", "exclusive",
    "report", "study", "analysis", "finds", "shows", "reveals", "claims",
    "alleges", "says", "source", "sources", "according", "to", "a", "an",
    "this", "that", "these", "those", "here", "there", "when", "where",
    "why", "how", "what", "which", "who", "whom", "whose", "it", "its",
    "them", "they", "their", "their", "him", "his", "her", "she", "he",
    "you", "your", "we", "our", "us", "i", "my", "me", "than", "then",
    "also", "just", "only", "even", "such", "much", "many", "more", "most",
    "some", "any", "no", "not", "only", "own", "same", "so", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now"
])

# Regex for common date patterns (expandable)
# Covers YYYY, Month DD, Month YYYY, DD Month YYYY, DD/MM/YYYY etc.
DATE_PATTERNS = re.compile(
    r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|'
    r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b|' # Month DD, Month DD, YYYY
    r'\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|'
    r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:,\s*\d{4})?\b|' # DD Month
    r'\b\d{4}\b|' # YYYY (general year)
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b' # DD/MM/YYYY or DD-MM-YYYY
    , re.IGNORECASE
)

class TemporalAnalyzer:
    """
    Analyze temporal patterns and detect recycled content (old news resurfacing).
    Enhanced with sophisticated signature generation, history tracking, and date parsing.
    """
    
    def __init__(self, history_retention_days: int = 365, max_history_size: int = 10000):
        # Stores {story_signature: [{'first_seen': timestamp, 'last_seen': timestamp, 'mentions_count': int, 'extracted_dates': List[datetime]}]}
        self.story_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Could store {recycled_pattern_id: {'headlines': List[str], 'first_detection': timestamp, 'last_detection': timestamp, 'recurrence_count': int}}
        # For now, it remains unused to strictly adhere to existing API, but could be integrated later.
        self.recycled_patterns: Dict = {} # Keeping this for API consistency, but not actively used for now.

        self.history_retention_days = history_retention_days
        self.max_history_size = max_history_size
        self._cleanup_threshold_time = datetime.now() - timedelta(days=self.history_retention_days)

        logger.info(f"TemporalAnalyzer initialized. History retention: {history_retention_days} days.")

    @lru_cache(maxsize=5000)
    def _generate_story_signature(self, headline: str) -> str:
        """
        Generate a more robust story signature based on normalized key terms.
        Uses a hash of sorted, cleaned, and stemmed/lemmatized (if NLTK/SpaCy were used) terms.
        For now, just cleaning and sorting.
        """
        # Lowercase, remove punctuation and extra spaces
        cleaned = re.sub(r'[^\w\s]', '', headline.lower())
        cleaned = ' '.join(cleaned.split())

        # Remove common stopwords (expand STOPWORDS for better accuracy)
        words = [word for word in cleaned.split() if word not in STOPWORDS and len(word) > 2]
        
        # Sort words to ensure consistent signature regardless of word order
        key_words = sorted(words)
        
        # Hash a representative subset (e.g., first 10, or all if short)
        signature_input = ' '.join(key_words[:10]) if len(key_words) > 10 else ' '.join(key_words)
        
        if not signature_input: # Handle cases where headline is all stopwords or too short
            signature_input = headline.lower() # Fallback to full headline
            
        return hashlib.md5(signature_input.encode('utf-8')).hexdigest()[:12] # Longer hash for robustness

    @lru_cache(maxsize=1000)
    def _extract_and_parse_dates(self, text: str) -> List[datetime]:
        """
        Extracts and parses common date patterns from text.
        More robust than just year extraction.
        """
        found_dates = []
        matches = DATE_PATTERNS.findall(text)
        current_year = datetime.now().year

        for match in matches:
            # Handle group matches if regex has multiple capturing groups
            if isinstance(match, tuple):
                match_str = ''.join(filter(None, match)) # Combine non-empty groups
            else:
                match_str = match

            try:
                # Try common formats first
                for fmt in ["%B %d, %Y", "%b %d, %Y", "%d %B, %Y", "%d %b, %Y", "%Y", "%m/%d/%Y", "%d/%m/%Y"]:
                    # Handle year-only case specially
                    if fmt == "%Y" and re.fullmatch(r'\d{4}', match_str):
                        year_val = int(match_str)
                        # Avoid future dates and very old ones (e.g., before 1900)
                        if 1900 <= year_val <= current_year + 1: # Allow current/next year in case of future projections
                            found_dates.append(datetime(year_val, 1, 1)) # Default to Jan 1
                            break
                    elif fmt == "%m/%d/%Y" or fmt == "%d/%m/%Y":
                        # Be careful with month/day ambiguity without explicitly checking order
                        # For simplicity, if it matches, try to parse
                        if re.fullmatch(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', match_str):
                            if len(match_str.split('/')[2]) == 2: # Check for 2-digit year
                                # Assume 20xx for years up to current_year+10, 19xx for others
                                # This is a common heuristic but can be wrong
                                parts = match_str.split('/')
                                if len(parts[2]) == 2:
                                    year_prefix = '20' if int(parts[2]) <= (current_year % 100) + 10 else '19'
                                    match_str = parts[0] + '/' + parts[1] + '/' + year_prefix + parts[2]
                                    fmt = "%m/%d/%Y" # Or whichever order is assumed for your region
                                    try:
                                        found_dates.append(datetime.strptime(match_str, fmt))
                                        break
                                    except ValueError:
                                        fmt = "%d/%m/%Y" # Try other order
                                        found_dates.append(datetime.strptime(match_str, fmt))
                                        break
                            else: # 4-digit year already
                                found_dates.append(datetime.strptime(match_str, fmt))
                                break
                    else: # For full month names
                        try:
                            parsed_date = datetime.strptime(match_str, fmt)
                            # If year is missing in format, assume current year
                            if '%Y' not in fmt:
                                parsed_date = parsed_date.replace(year=current_year)
                            found_dates.append(parsed_date)
                            break
                        except ValueError:
                            continue # Try next format
            except ValueError:
                continue # Skip unparsable dates

        # Filter out future dates and very implausible past dates
        now = datetime.now()
        found_dates = [d for d in found_dates if d <= now and d.year >= 1900]
        
        return found_dates

    def check_recycled_content(self, headline: str) -> Dict:
        """
        Check if this is recycled news by:
        1. Looking for outdated date references in the headline.
        2. Comparing the headline signature against historical records.
        """
        story_signature = self._generate_story_signature(headline)
        current_timestamp = datetime.now()
        
        recycled_score = 0.0 # Default to no recycling detected
        recycled_reason = []
        
        # --- Heuristic 1: Outdated Date References in Headline ---
        extracted_dates = self._extract_and_parse_dates(headline)
        if extracted_dates:
            oldest_headline_date = min(extracted_dates)
            # If the oldest date mentioned in the headline is more than X years old
            # AND it's not clearly an historical event (e.g., "Battle of Hastings 1066")
            # For simplicity, penalize if it's too old relative to current time.
            age_in_years = (current_timestamp - oldest_headline_date).days / 365.25
            
            if age_in_years > 3: # Example: Headline refers to something 3+ years ago
                recycled_score += 0.3
                recycled_reason.append(f"Outdated date reference in headline: {oldest_headline_date.year}")
            elif age_in_years > 1:
                 recycled_score += 0.1
                 recycled_reason.append(f"Older date reference in headline: {oldest_headline_date.year}")

        # --- Heuristic 2: Check against Internal Story History ---
        # Update history and check for past occurrences
        self._update_story_history(story_signature, current_timestamp, extracted_dates)

        # Check if the story has been seen before and how often/recently
        history_entries = self.story_history[story_signature]
        if len(history_entries) > 1: # Has been seen multiple times
            # Sort by last_seen to find recurrence
            history_entries.sort(key=lambda x: x['last_seen'], reverse=True)
            most_recent_occurrence = history_entries[0]
            second_most_recent_occurrence = history_entries[1] if len(history_entries) > 1 else None

            # Calculate time since last appearance
            time_since_last_seen = current_timestamp - most_recent_occurrence['last_seen']
            
            # Simple check for rapid re-appearance (could be organic, or orchestrated)
            if time_since_last_seen < timedelta(days=7) and most_recent_occurrence['mentions_count'] > 2:
                # Could be a trending topic, but also rapid recycling
                pass # For now, don't penalize rapid re-appearance too much without more context

            # Significant gap between appearances indicates recycling
            if second_most_recent_occurrence:
                gap = most_recent_occurrence['first_seen'] - second_most_recent_occurrence['last_seen']
                if gap > timedelta(days=90): # If it reappears after 3 months
                    recycled_score += 0.4
                    recycled_reason.append(f"Significant temporal gap between occurrences ({gap.days} days).")
                elif gap > timedelta(days=30): # Reappears after 1 month
                    recycled_score += 0.2
                    recycled_reason.append(f"Noticeable temporal gap between occurrences ({gap.days} days).")
            
            # If overall mentions count is high and spread over a long period
            total_mentions = sum(entry['mentions_count'] for entry in history_entries)
            oldest_first_seen = min(entry['first_seen'] for entry in history_entries)
            overall_age = current_timestamp - oldest_first_seen
            if total_mentions > 5 and overall_age > timedelta(days=365):
                recycled_score += 0.2
                recycled_reason.append(f"Frequent recurrence over long period ({overall_age.days} days).")
            
            # Penalize if headline refers to historical event but presented as new without clear context
            # (Requires NLP to detect "historical event" context - simplified here)
            if age_in_years > 5 and 'recycled_pattern' not in recycled_reason and not any(kw in headline.lower() for kw in ["history of", "historical perspective", "looking back at", "anniversary"]):
                recycled_score += 0.1 # Small penalty if very old but no explicit historical context


        # Clamp the score to [0.0, 1.0]
        final_recycled_score = max(0.0, min(1.0, recycled_score))

        logger.info(f"Headline: '{headline[:50]}...' | Signature: {story_signature} | Recycled Score: {final_recycled_score:.3f}")
        logger.debug(f"Recycled reasons: {recycled_reason}")

        return {
            "score": final_recycled_score,
            "signature": story_signature,
            "potential_recycled": final_recycled_score > 0.2, # Threshold can be tuned
            "reasons": recycled_reason # New: Provide reasons for the score
        }

    def _update_story_history(self, story_signature: str, timestamp: datetime, extracted_dates: List[datetime]):
        """
        Updates the internal history for a given story signature.
        Manages entries to prevent excessive growth.
        """
        # Find if there's a recent entry for this signature
        # We assume multiple calls to check_recycled_content for the same 'event' will happen close in time
        updated = False
        for entry in self.story_history[story_signature]:
            # If this entry was seen recently (e.g., within 24 hours), update it
            if timestamp - entry['last_seen'] < timedelta(hours=24):
                entry['last_seen'] = timestamp
                entry['mentions_count'] += 1
                entry['extracted_dates'].extend(d for d in extracted_dates if d not in entry['extracted_dates'])
                updated = True
                break
        
        if not updated:
            # Add a new entry for this occurrence
            self.story_history[story_signature].append({
                'first_seen': timestamp,
                'last_seen': timestamp,
                'mentions_count': 1,
                'extracted_dates': extracted_dates
            })
            # Ensure the list of entries for a signature doesn't grow indefinitely if distinct occurrences are logged
            # You might want to refine this to only keep a few distinct "recurrences"
            # For simplicity, we just add, and rely on overall cleanup.

        # Periodically clean up old entries to prevent memory bloat
        # This cleanup happens only when a new story is processed or an existing one is updated,
        # which is more efficient than a separate timer thread.
        self._periodic_history_cleanup()
        
    def _periodic_history_cleanup(self):
        """Removes very old story entries or prunes history if size limit exceeded."""
        # Clean up old entries within each story signature's list
        for signature in list(self.story_history.keys()): # Iterate over a copy of keys
            self.story_history[signature] = [
                entry for entry in self.story_history[signature] 
                if entry['last_seen'] >= self._cleanup_threshold_time
            ]
            if not self.story_history[signature]:
                del self.story_history[signature] # Remove signature if all entries are old

        # If overall history size grows too large, remove oldest stories based on first_seen
        if len(self.story_history) > self.max_history_size:
            logger.warning(f"TemporalAnalyzer history size ({len(self.story_history)}) exceeds max_history_size ({self.max_history_size}). Pruning oldest entries.")
            sorted_stories = sorted(self.story_history.items(), key=lambda item: min(e['first_seen'] for e in item[1]))
            num_to_prune = len(self.story_history) - self.max_history_size + int(self.max_history_size * 0.1) # Prune 10% more than overflow
            for i in range(num_to_prune):
                del self.story_history[sorted_stories[i][0]]

        # Update the cleanup threshold to reflect current time for next pass
        self._cleanup_threshold_time = datetime.now() - timedelta(days=self.history_retention_days)


# Example Usage
if __name__ == "__main__":
    analyzer = TemporalAnalyzer(history_retention_days=30, max_history_size=100)

    print("--- Temporal Analysis Examples ---")

    # Scenario 1: Fresh News (no dates, first time seen)
    headline1 = "Scientists discover new exoplanet in nearby galaxy."
    result1 = analyzer.check_recycled_content(headline1)
    print(f"\nHeadline: '{headline1}'")
    print(f"Result: {result1}")

    # Scenario 2: News with an old date reference
    headline2 = "Rare fish spotted in Lake Victoria in 1998, expert says."
    result2 = analyzer.check_recycled_content(headline2)
    print(f"\nHeadline: '{headline2}'")
    print(f"Result: {result2}")

    # Scenario 3: Recycled Headline (same signature reappears after some time)
    headline3_v1 = "Government unveils new economic policy plan."
    result3_v1 = analyzer.check_recycled_content(headline3_v1)
    print(f"\nHeadline: '{headline3_v1}'")
    print(f"Result (first seen): {result3_v1}")

    # Simulate time passing or a different context call
    import time
    print("Simulating 3 days passing...")
    time.sleep(3) # Simulate 3-day gap (for demo purposes, not actual time)
    # To truly simulate a gap, you'd need to mock datetime.now() or set a fixed time for testing
    
    # Scenario 3.1: Same headline reappears shortly after (not necessarily recycled)
    headline3_v2 = "Government unveils new economic policy plan." # Same headline
    result3_v2 = analyzer.check_recycled_content(headline3_v2)
    print(f"\nHeadline: '{headline3_v2}'")
    print(f"Result (seen again shortly): {result3_v2}")

    print("Simulating 95 days passing for significant gap detection...")
    # Manually adjust a history entry to simulate a large time gap for testing
    signature3 = analyzer._generate_story_signature(headline3_v1)
    if signature3 in analyzer.story_history and analyzer.story_history[signature3]:
        # Find the specific entry for this signature (assuming it's the first one, or adjust if multiple)
        # We'll modify the 'last_seen' of the most recent entry to simulate it being much older.
        analyzer.story_history[signature3][-1]['last_seen'] = datetime.now() - timedelta(days=95)
        analyzer.story_history[signature3][-1]['first_seen'] = datetime.now() - timedelta(days=95) - timedelta(days=5) # Also make first seen old
        print(f"Manually adjusted '{headline3_v1}' entry for a large gap test.")

    # Scenario 3.2: Same headline reappears after a significant gap
    headline3_v3 = "Government unveils new economic policy plan." # Same headline
    result3_v3 = analyzer.check_recycled_content(headline3_v3)
    print(f"\nHeadline: '{headline3_v3}'")
    print(f"Result (seen after significant gap): {result3_v3}")
    
    # Scenario 4: Headline with a current year reference (should not be recycled)
    headline4 = "New scientific discovery announced in 2025."
    result4 = analyzer.check_recycled_content(headline4)
    print(f"\nHeadline: '{headline4}'")
    print(f"Result: {result4}")

    # Scenario 5: Multiple historical dates
    headline5 = "The Battle of Waterloo (1815) and its impact on Europe in 1820."
    result5 = analyzer.check_recycled_content(headline5)
    print(f"\nHeadline: '{headline5}'")
    print(f"Result: {result5}")

    # Test the cleanup mechanism (requires many unique headlines or longer run)
    print("\n--- Testing Cleanup (Spamming Headlines) ---")
    for i in range(150): # Assuming max_history_size is 100
        test_h = f"Unique news {i} for testing cleanup {datetime.now().isoformat()}"
        analyzer.check_recycled_content(test_h)
    print(f"Current history size: {len(analyzer.story_history)}")

    # Check a specific story's history (internal check)
    if signature3 in analyzer.story_history:
        print(f"\nHistory for signature '{signature3}':")
        for entry in analyzer.story_history[signature3]:
            print(f"  First: {entry['first_seen'].strftime('%Y-%m-%d %H:%M:%S')}, Last: {entry['last_seen'].strftime('%Y-%m-%d %H:%M:%S')}, Mentions: {entry['mentions_count']}")