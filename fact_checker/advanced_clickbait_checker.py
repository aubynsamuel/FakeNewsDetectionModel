import os
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


class HybridClickbaitDetector:
    def __init__(self):
        self.classifier = None
        self.tfidf_vectorizer = None
        self.is_trained = False

        # Enhanced clickbait patterns with semantic categories
        self.clickbait_indicators = {
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

    def extract_enhanced_features(self, texts):
        """Extract comprehensive handcrafted features"""
        features = []

        for text in texts:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            text_lower = text.lower()
            feature_vector = []

            # Clickbait pattern scores by category
            for category, patterns in self.clickbait_indicators.items():
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
            all_caps_words = sum(
                1 for word in words if word.isupper() and len(word) > 1
            )
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

    def load_and_prepare_data(self):
        """Load and prepare training data"""
        print("Loading datasets...")

        # Load your existing data files
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(current_script_dir)
        clickbait_csv_path = os.path.join(
            project_root_dir, "data", "fixed_clickbait_data.csv"
        )
        non_clickbait_csv_path = os.path.join(
            project_root_dir, "data", "fixed_non_clickbait_data.csv"
        )

        try:
            # Load clickbait data
            clickbait_df = pd.read_csv(clickbait_csv_path, header=None, engine="python")
            clickbait_texts = clickbait_df.iloc[:, 0].astype(str).tolist()
            clickbait_labels = [1] * len(clickbait_texts)

            # Load non-clickbait data
            non_clickbait_df = pd.read_csv(
                non_clickbait_csv_path, header=None, engine="python"
            )
            non_clickbait_texts = non_clickbait_df.iloc[:, 0].astype(str).tolist()
            non_clickbait_labels = [0] * len(non_clickbait_texts)

            # Combine and shuffle
            all_texts = clickbait_texts + non_clickbait_texts
            all_labels = clickbait_labels + non_clickbait_labels

            print(f"Loaded {len(all_texts)} total samples")
            print(f"Clickbait ratio: {sum(all_labels) / len(all_labels):.2%}")

            return all_texts, all_labels

        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def train(
        self,
        output_dir="./optimized_model",
        epochs=10,  # Increased epochs for Logistic Regression
        batch_size=None,  # Not used for Logistic Regression
        learning_rate=None,  # Not used for Logistic Regression
    ):
        """Train the optimized hybrid model"""
        print("Starting optimized training...")

        # Load data
        texts, labels = self.load_and_prepare_data()
        if texts is None:
            return False

        # Clean and filter data
        valid_data = [
            (text, label)
            for text, label in zip(texts, labels)
            if text and len(text.strip()) > 5
        ]
        texts, labels = list(zip(*valid_data)) if valid_data else ([], [])

        print(f"Training with {len(texts)} samples")

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels
        )

        print("Training TF-IDF vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Adjusted for potentially better performance
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            stop_words="english",
        )
        train_tfidf_features = self.tfidf_vectorizer.fit_transform(train_texts)
        val_tfidf_features = self.tfidf_vectorizer.transform(val_texts)
        print("TF-IDF vectorizer trained.")

        print("Extracting enhanced features...")
        train_handcrafted_features = self.extract_enhanced_features(train_texts)
        val_handcrafted_features = self.extract_enhanced_features(val_texts)
        print("Enhanced features extracted.")

        # Combine features
        X_train = np.hstack(
            (train_tfidf_features.toarray(), train_handcrafted_features)
        )
        X_val = np.hstack((val_tfidf_features.toarray(), val_handcrafted_features))
        y_train = np.array(train_labels)
        y_val = np.array(val_labels)

        print("Initializing and training Logistic Regression model...")
        self.classifier = LogisticRegression(
            max_iter=epochs, solver="liblinear", random_state=42, verbose=1
        )
        self.classifier.fit(X_train, y_train)
        print("Logistic Regression model trained.")

        # Evaluate model
        print("Evaluating model...")
        predictions = self.classifier.predict(X_val)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, predictions, average="binary"
        )
        accuracy = accuracy_score(y_val, predictions)

        print(f"\nValidation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        # Save the trained model and vectorizer
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/logistic_regression_model.pkl", "wb") as f:
            pickle.dump(self.classifier, f)
        with open(f"{output_dir}/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open(f"{output_dir}/feature_info.pkl", "wb") as f:
            pickle.dump(self.clickbait_indicators, f)

        self.is_trained = True
        print("Model and vectorizer saved.")

        return True

    def load_model(self, model_dir="optimized_model"):
        """Load the trained model"""
        try:
            with open(f"{model_dir}/logistic_regression_model.pkl", "rb") as f:
                self.classifier = pickle.load(f)

            # Load TF-IDF vectorizer
            with open(f"{model_dir}/tfidf_vectorizer.pkl", "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)

            # Load feature info
            with open(f"{model_dir}/feature_info.pkl", "rb") as f:
                self.clickbait_indicators = pickle.load(f)

            self.is_trained = True
            print("Model loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict_clickbait(self, headline, threshold=0.5):
        """Predict clickbait with hybrid approach"""
        if not self.is_trained:
            if not self.load_model():
                return False, 0.0, 0.0

        # TF-IDF + handcrafted features score
        tfidf_features = self.tfidf_vectorizer.transform([headline])
        handcrafted_features = self.extract_enhanced_features([headline])

        # Combine features
        combined_features = np.hstack((tfidf_features.toarray(), handcrafted_features))

        # Logistic Regression prediction
        lr_probs = self.classifier.predict_proba(combined_features)[0]
        lr_score = lr_probs[1]  # Probability of being clickbait

        is_clickbait = lr_score >= threshold
        confidence = lr_score if is_clickbait else (1 - lr_score)

        return is_clickbait, lr_score, confidence


# Usage example
if __name__ == "__main__":
    detector = HybridClickbaitDetector()
    while True:
        headline = input("Enter a headline to check clickbait score: ")
        is_clickbait, score, confidence = detector.predict_clickbait(headline)
        status = "CLICKBAIT" if is_clickbait else "NORMAL"
        print(f"{status} (Score: {score:.3f}, Confidence: {confidence:.3f})")
        print(f"  '{headline}'")
        print()

    # Train the model automatically for testing
    # success = detector.train(epochs=200)  # Use default epochs for LR

    # if success:
    #     # Test the model
    #     test_headlines = [
    #         "You Won't Believe What Happened Next!",
    #         "Scientists discover new treatment for diabetes",
    #         "10 Amazing Secrets That Will Change Your Life Forever",
    #         "Economic growth shows steady improvement in Q3",
    #         "This Simple Trick Will Shock You - Doctors Hate It!",
    #         "Research published on climate change impacts",
    #     ]

    #     print("\nTesting hybrid model:")
    #     print("-" * 60)

    #     for headline in test_headlines:
    #         is_clickbait, score, confidence = detector.predict_clickbait(headline)
    #         status = "CLICKBAIT" if is_clickbait else "NORMAL"
    #         print(f"{status} (Score: {score:.3f}, Confidence: {confidence:.3f})")
    #         print(f"  '{headline}'")
    #         print()
