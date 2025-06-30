import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

class MLClickbaitDetector:
    """ML-based clickbait and sensational language detector"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # Enhanced suspicious patterns
        self.suspicious_patterns = [
            r'\b(shocking|unbelievable|incredible|amazing|stunning)\b',
            r'\b(\d+\s+tricks?|one\s+trick|simple\s+trick)\b',
            r'\b(doctors|scientists|experts)\s+(hate|don\'t\s+want|shocked)\b',
            r'\b(breaking|urgent|alert|warning)[:!]\s*\b',
            r'\b(leaked|hidden|secret|exposed|revealed)\b',
            r'\b(miracle|instant|amazing|incredible)\s+(cure|results?|discovery)\b',
            r'\b(you\s+won\'t\s+believe|wait\s+until\s+you\s+see)\b',
            r'\b(this\s+will\s+blow\s+your\s+mind|mind\s*=\s*blown)\b',
            r'^\s*\d+\s+(reasons?|ways?|things?|facts?)\s+',
            r'\b(gone\s+wrong|gone\s+viral|Internet\s+is\s+going\s+crazy)\b',
        ]
        
    def create_training_data(self):
        """Create synthetic training data for clickbait detection"""
        clickbait_examples = [
            "You Won't Believe What Happened Next!",
            "Doctors Hate This One Simple Trick",
            "SHOCKING: Scientists Don't Want You to Know This",
            "7 Amazing Facts That Will Blow Your Mind",
            "This Miracle Cure Will Change Your Life Forever",
            "BREAKING: Leaked Footage Shows Hidden Truth",
            "Wait Until You See What Happens Next",
            "This Simple Trick Will Save You Thousands",
            "Scientists Are Baffled by This Discovery",
            "The Internet Is Going Crazy Over This"
        ]
        
        normal_examples = [
            "Parliament passes new healthcare legislation",
            "Economic growth shows signs of recovery",
            "Research study reveals climate change impacts",
            "Technology company announces quarterly results",
            "Local community organizes charity event",
            "University researchers publish new findings",
            "Government announces infrastructure investment",
            "Sports team wins championship final",
            "Weather forecast predicts rainfall this week",
            "Market analysis shows steady performance"
        ]
        
        texts = clickbait_examples + normal_examples
        labels = [1] * len(clickbait_examples) + [0] * len(normal_examples)
        
        return texts, labels
    
    def train(self):
        """Train the clickbait detector"""
        texts, labels = self.create_training_data()
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def detect_clickbait_score(self, headline: str) -> float:
        """Calculate clickbait score using both ML and pattern matching"""
        if not self.is_trained:
            self.train()
        
        # Pattern-based score
        pattern_score = 0
        headline_lower = headline.lower()
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, headline_lower):
                pattern_score += 0.2
        
        # Punctuation and capitalization analysis
        exclamation_count = headline.count('!')
        caps_ratio = sum(1 for c in headline if c.isupper()) / len(headline) if headline else 0
        
        pattern_score += min(exclamation_count * 0.1, 0.3)
        pattern_score += caps_ratio * 0.2
        
        # ML-based score
        try:
            X = self.vectorizer.transform([headline])
            ml_score = self.classifier.predict_proba(X)[0][1]  # Probability of being clickbait
        except:
            ml_score = 0.0
        
        # Combine scores (weighted average)
        final_score = (pattern_score * 0.4 + ml_score * 0.6)
        return min(final_score, 1.0)