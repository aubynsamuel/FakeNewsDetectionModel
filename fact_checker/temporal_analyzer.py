import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from collections import defaultdict
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(" %(message)s"))
    logger.addHandler(ch)

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
        self.story_history: Dict[str, List[Dict]] = defaultdict(list)
        
        self.recycled_patterns: Dict = {}

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
        cleaned = re.sub(r'[^\w\s]', '', headline.lower())
        cleaned = ' '.join(cleaned.split())

        words = [word for word in cleaned.split() if word not in STOPWORDS and len(word) > 2]
        
        key_words = sorted(words)
        
        signature_input = ' '.join(key_words[:10]) if len(key_words) > 10 else ' '.join(key_words)
        
        if not signature_input:
            signature_input = headline.lower() 
            
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
            if isinstance(match, tuple):
                match_str = ''.join(filter(None, match))
            else:
                match_str = match

            try:
                for fmt in ["%B %d, %Y", "%b %d, %Y", "%d %B, %Y", "%d %b, %Y", "%Y", "%m/%d/%Y", "%d/%m/%Y"]:
                    if fmt == "%Y" and re.fullmatch(r'\d{4}', match_str):
                        year_val = int(match_str)
                        if 1900 <= year_val <= current_year + 1:
                            found_dates.append(datetime(year_val, 1, 1))
                            break
                    elif fmt == "%m/%d/%Y" or fmt == "%d/%m/%Y":
                        if re.fullmatch(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', match_str):
                            if len(match_str.split('/')[2]) == 2:
                                parts = match_str.split('/')
                                if len(parts[2]) == 2:
                                    year_prefix = '20' if int(parts[2]) <= (current_year % 100) + 10 else '19'
                                    match_str = parts[0] + '/' + parts[1] + '/' + year_prefix + parts[2]
                                    fmt = "%m/%d/%Y"
                                    try:
                                        found_dates.append(datetime.strptime(match_str, fmt))
                                        break
                                    except ValueError:
                                        fmt = "%d/%m/%Y"
                                        found_dates.append(datetime.strptime(match_str, fmt))
                                        break
                            else:
                                found_dates.append(datetime.strptime(match_str, fmt))
                                break
                    else:
                        try:
                            parsed_date = datetime.strptime(match_str, fmt)
                            if '%Y' not in fmt:
                                parsed_date = parsed_date.replace(year=current_year)
                            found_dates.append(parsed_date)
                            break
                        except ValueError:
                            continue
            except ValueError:
                continue

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
        
        recycled_score = 0.0
        recycled_reason = []
        
        extracted_dates = self._extract_and_parse_dates(headline)
        if extracted_dates:
            oldest_headline_date = min(extracted_dates)
          
            age_in_years = (current_timestamp - oldest_headline_date).days / 365.25
            
            if age_in_years > 3:
                recycled_score += 0.3
                recycled_reason.append(f"Outdated date reference in headline: {oldest_headline_date.year}")
            elif age_in_years > 1:
                 recycled_score += 0.1
                 recycled_reason.append(f"Older date reference in headline: {oldest_headline_date.year}")

        
        self._update_story_history(story_signature, current_timestamp, extracted_dates)

        history_entries = self.story_history[story_signature]
        if len(history_entries) > 1:
            history_entries.sort(key=lambda x: x['last_seen'], reverse=True)
            most_recent_occurrence = history_entries[0]
            second_most_recent_occurrence = history_entries[1] if len(history_entries) > 1 else None

            time_since_last_seen = current_timestamp - most_recent_occurrence['last_seen']
            
            if time_since_last_seen < timedelta(days=7) and most_recent_occurrence['mentions_count'] > 2:
                pass 

            if second_most_recent_occurrence:
                gap = most_recent_occurrence['first_seen'] - second_most_recent_occurrence['last_seen']
                if gap > timedelta(days=90): 
                    recycled_score += 0.4
                    recycled_reason.append(f"Significant temporal gap between occurrences ({gap.days} days).")
                elif gap > timedelta(days=30): 
                    recycled_score += 0.2
                    recycled_reason.append(f"Noticeable temporal gap between occurrences ({gap.days} days).")
            
            total_mentions = sum(entry['mentions_count'] for entry in history_entries)
            oldest_first_seen = min(entry['first_seen'] for entry in history_entries)
            overall_age = current_timestamp - oldest_first_seen
            if total_mentions > 5 and overall_age > timedelta(days=365):
                recycled_score += 0.2
                recycled_reason.append(f"Frequent recurrence over long period ({overall_age.days} days).")
            
          
            if age_in_years > 5 and 'recycled_pattern' not in recycled_reason and not any(kw in headline.lower() for kw in ["history of", "historical perspective", "looking back at", "anniversary"]):
                recycled_score += 0.1 


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
        updated = False
        for entry in self.story_history[story_signature]:
            if timestamp - entry['last_seen'] < timedelta(hours=24):
                entry['last_seen'] = timestamp
                entry['mentions_count'] += 1
                entry['extracted_dates'].extend(d for d in extracted_dates if d not in entry['extracted_dates'])
                updated = True
                break
        
        if not updated:
            self.story_history[story_signature].append({
                'first_seen': timestamp,
                'last_seen': timestamp,
                'mentions_count': 1,
                'extracted_dates': extracted_dates
            })
            
        self._periodic_history_cleanup()
        
    def _periodic_history_cleanup(self):
        """Removes very old story entries or prunes history if size limit exceeded."""
        for signature in list(self.story_history.keys()): 
            self.story_history[signature] = [
                entry for entry in self.story_history[signature] 
                if entry['last_seen'] >= self._cleanup_threshold_time
            ]
            if not self.story_history[signature]:
                del self.story_history[signature]

        if len(self.story_history) > self.max_history_size:
            logger.warning(f"TemporalAnalyzer history size ({len(self.story_history)}) exceeds max_history_size ({self.max_history_size}). Pruning oldest entries.")
            sorted_stories = sorted(self.story_history.items(), key=lambda item: min(e['first_seen'] for e in item[1]))
            num_to_prune = len(self.story_history) - self.max_history_size + int(self.max_history_size * 0.1) # Prune 10% more than overflow
            for i in range(num_to_prune):
                del self.story_history[sorted_stories[i][0]]

        # Update the cleanup threshold to reflect current time for next pass
        self._cleanup_threshold_time = datetime.now() - timedelta(days=self.history_retention_days)