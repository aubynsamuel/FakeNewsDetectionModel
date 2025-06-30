import re
import hashlib
from datetime import datetime
from typing import Dict

class TemporalAnalyzer:
    """Analyze temporal patterns and recycled content"""
    
    def __init__(self):
        self.story_history = {}
        self.recycled_patterns = {}
    
    def check_recycled_content(self, headline: str) -> Dict:
        """Check if this is recycled fake news"""
        story_signature = self._generate_story_signature(headline)
        
        # For now, return basic analysis
        # In production, this would check against a database of known fake stories
        recycled_score = 0.0
        
        # Simple heuristic: check for date references that seem outdated
        current_year = datetime.now().year
        years_in_headline = re.findall(r'\b(19|20)\d{2}\b', headline)
        
        if years_in_headline:
            oldest_year = min(int(year) for year in years_in_headline)
            if current_year - oldest_year > 2:
                recycled_score += 0.3
        
        return {
            "score": recycled_score,
            "signature": story_signature,
            "potential_recycled": recycled_score > 0.2
        }
    
    def _generate_story_signature(self, headline: str) -> str:
        """Generate a signature for story tracking"""
        # Remove dates, numbers, and create a hash of key terms
        cleaned = re.sub(r'\b\d+\b', '', headline.lower())
        words = cleaned.split()
        key_words = [word for word in words if len(word) > 3][:5]  # Top 5 meaningful words
        return hashlib.md5(' '.join(sorted(key_words)).encode()).hexdigest()[:8]