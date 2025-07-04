import re
import numpy as np

clickbait_indicators = {
    "curiosity_gap": [
        "you won't believe",
        "wait until you see",
        "what happened next",
        "the reason will shock you",
        "this is why",
        "here's what happened",
        "the truth about",
        "what nobody tells you",
        "finally revealed",
    ],
    "emotional_triggers": [
        "shocking",
        "incredible",
        "amazing",
        "unbelievable",
        "stunning",
        "heartbreaking",
        "hilarious",
        "terrifying",
        "adorable",
        "outrageous",
        "mind-blowing",
        "jaw-dropping",
        "breathtaking",
    ],
    "urgency_scarcity": [
        "breaking",
        "urgent",
        "limited time",
        "before it's too late",
        "act now",
        "don't miss",
        "last chance",
        "expires soon",
    ],
    "personal_relevance": [
        "in your area",
        "people like you",
        "your age",
        "based on your",
        "you need to know",
        "this affects you",
        "for people who",
    ],
    "superlatives": [
        "ultimate",
        "perfect",
        "best ever",
        "greatest",
        "worst",
        "most amazing",
        "incredible",
        "unmatched",
        "revolutionary",
    ],
    "numbers_lists": [
        r"\d+\s+(reasons?|ways?|things?|facts?|secrets?|tricks?|tips?)",
        r"one\s+(weird|simple|amazing)\s+trick",
        r"\d+\s+minute[s]?",
        r"in\s+\d+\s+(steps?|minutes?|days?)",
    ],
    "authority_social_proof": [
        "doctors hate",
        "experts don't want",
        "celebrities use",
        "scientists discovered",
        "research shows",
        "studies prove",
    ],
}


def extract_enhanced_features(texts):
    """Extract comprehensive handcrafted features"""
    features = []

    for text in texts:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        text_lower = text.lower()
        feature_vector = []

        # Clickbait pattern scores by category
        for category, patterns in clickbait_indicators.items():
            category_score = 0
            for pattern in patterns:
                if isinstance(pattern, str):
                    if pattern in text_lower:
                        category_score += 1
                else:  # regex pattern
                    if re.search(pattern, text_lower):
                        category_score += 1

            # Normalize by pattern count in category
            normalized_score = min(category_score / len(patterns), 1.0)
            feature_vector.append(normalized_score)

        # Punctuation and formatting features
        exclamation_ratio = text.count("!") / max(len(text), 1)
        question_ratio = text.count("?") / max(len(text), 1)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        feature_vector.extend(
            [
                min(exclamation_ratio * 10, 1.0),
                min(question_ratio * 10, 1.0),
                min(caps_ratio * 5, 1.0),
            ]
        )

        # Length and structure features
        words = text.split()
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)

        feature_vector.extend(
            [
                min(word_count / 20, 1.0),  # Normalized word count
                min(avg_word_length / 8, 1.0),  # Normalized avg word length
                1.0 if word_count > 10 else 0.0,  # Long headline indicator
            ]
        )

        # Semantic features
        all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        number_count = len(
            [word for word in words if any(char.isdigit() for char in word)]
        )

        feature_vector.extend(
            [
                min(all_caps_words / max(word_count, 1), 1.0),
                min(number_count / max(word_count, 1), 1.0),
            ]
        )

        features.append(feature_vector)

    return np.array(features)
