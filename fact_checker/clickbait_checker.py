from pathlib import Path
import re
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import VotingClassifier
import warnings

warnings.filterwarnings("ignore")

from utils import parallel_preprocess


class AdvancedClickbaitDetector:
    """Advanced ML-based clickbait detector with ensemble methods and feature engineering"""

    def __init__(self):
        self.vectorizer = None
        self.ensemble_model = None
        self.is_trained = False
        self.feature_names = []

        # Enhanced suspicious patterns with weights
        self.suspicious_patterns = [
            (r"\b(shocking|unbelievable|incredible|amazing|stunning)\b", 0.3),
            (r"\b(\d+\s+tricks?|one\s+trick|simple\s+trick)\b", 0.4),
            (r"\b(doctors|scientists|experts)\s+(hate|don\'t\s+want|shocked)\b", 0.5),
            (r"\b(breaking|urgent|alert|warning)[:!]\s*\b", 0.2),
            (r"\b(leaked|hidden|secret|exposed|revealed)\b", 0.3),
            (
                r"\b(miracle|instant|amazing|incredible)\s+(cure|results?|discovery)\b",
                0.4,
            ),
            (r"\b(you\s+won\'t\s+believe|wait\s+until\s+you\s+see)\b", 0.5),
            (r"\b(this\s+will\s+blow\s+your\s+mind|mind\s*=\s*blown)\b", 0.4),
            (r"^\s*\d+\s+(reasons?|ways?|things?|facts?)\s+", 0.3),
            (r"\b(gone\s+wrong|gone\s+viral|internet\s+is\s+going\s+crazy)\b", 0.3),
            (r"\b(must\s+see|must\s+watch|must\s+read)\b", 0.2),
            (r"\b(ultimate|perfect|best\s+ever|greatest)\b", 0.2),
            (r"\b(transform|transformation|change\s+your\s+life)\b", 0.3),
        ]

    def extract_handcrafted_features(self, texts):
        """Extract handcrafted features from texts"""
        features = []

        for text in texts:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            text_lower = text.lower()
            feature_vector = []

            # Pattern-based features
            pattern_score = 0
            for pattern, weight in self.suspicious_patterns:
                if re.search(pattern, text_lower):
                    pattern_score += weight
            feature_vector.append(min(pattern_score, 1.0))

            # Punctuation features
            exclamation_count = text.count("!")
            question_count = text.count("?")
            feature_vector.extend(
                [
                    min(exclamation_count / max(len(text.split()), 1), 1.0),
                    min(question_count / max(len(text.split()), 1), 1.0),
                ]
            )

            # Capitalization features
            if text:
                caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
                word_caps_ratio = sum(
                    1 for word in text.split() if word.isupper()
                ) / max(len(text.split()), 1)
            else:
                caps_ratio = word_caps_ratio = 0
            feature_vector.extend([caps_ratio, word_caps_ratio])

            # Length features
            word_count = len(text.split())
            char_count = len(text)
            avg_word_length = char_count / max(word_count, 1)
            feature_vector.extend(
                [
                    min(word_count / 20, 1.0),  # Normalized word count
                    min(char_count / 100, 1.0),  # Normalized char count
                    min(avg_word_length / 10, 1.0),  # Normalized avg word length
                ]
            )

            # Clickbait indicators
            clickbait_words = [
                "amazing",
                "shocking",
                "unbelievable",
                "incredible",
                "stunning",
                "secret",
                "hidden",
                "revealed",
                "exposed",
                "trick",
                "hack",
            ]
            clickbait_word_count = sum(
                1 for word in clickbait_words if word in text_lower
            )
            feature_vector.append(min(clickbait_word_count / max(word_count, 1), 1.0))

            # Number features
            number_count = len(re.findall(r"\d+", text))
            feature_vector.append(min(number_count / max(word_count, 1), 1.0))

            features.append(feature_vector)

        return np.array(features)

    def load_and_prepare_data(self):
        """Load and prepare training data from CSV files"""
        print("Loading datasets...")

        current_script_dir = Path(__file__).parent
        project_root_dir = current_script_dir.parent
        clickbait_csv_path = (
            project_root_dir / "data" / "fixed_clickbait_data.csv"
        )  # 16000 samples
        non_clickbait_csv_path = (
            project_root_dir / "data" / "fixed_non_clickbait_data.csv"
        )  # 16000 samples

        # Load clickbait data
        try:
            clickbait_df = pd.read_csv(clickbait_csv_path, header=None, engine="python")
            clickbait_texts = (
                clickbait_df.iloc[:, 0].astype(str).tolist()
            )  # First column
            clickbait_labels = [1] * len(clickbait_texts)
            print(f"Loaded {len(clickbait_texts)} clickbait samples")
        except Exception as e:
            print(f"Error loading clickbait data: {e}")
            return None, None

        # Load non-clickbait data
        try:
            non_clickbait_df = pd.read_csv(
                non_clickbait_csv_path, header=None, engine="python"
            )
            non_clickbait_texts = (
                non_clickbait_df.iloc[:, 0].astype(str).tolist()
            )  # First column
            non_clickbait_labels = [0] * len(non_clickbait_texts)
            print(f"Loaded {len(non_clickbait_texts)} non-clickbait samples")
        except Exception as e:
            print(f"Error loading non-clickbait data: {e}")
            return None, None

        # Combine datasets
        all_texts = clickbait_texts + non_clickbait_texts
        all_labels = clickbait_labels + non_clickbait_labels

        print(f"Total samples: {len(all_texts)}")
        print(f"Clickbait ratio: {sum(all_labels) / len(all_labels):.2%}")

        return all_texts, all_labels

    def train(self):
        """Train the advanced clickbait detector"""
        print("Starting training process...")

        # Load data
        texts, labels = self.load_and_prepare_data()
        if texts is None:
            print("Failed to load data. Cannot train model.")
            return False

        # Preprocess texts
        print("Preprocessing texts...")
        processed_texts = parallel_preprocess(texts)

        # Remove empty texts
        valid_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
        processed_texts = [processed_texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]

        print(f"Valid samples after preprocessing: {len(processed_texts)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Feature extraction
        print("Extracting features...")

        # TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True,
        )

        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Handcrafted features
        X_train_features = self.extract_handcrafted_features(X_train)
        X_test_features = self.extract_handcrafted_features(X_test)

        # Combine features
        from scipy.sparse import hstack, csr_matrix

        X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_features)])
        X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_features)])

        # Create ensemble model
        print("Training ensemble model...")

        # Individual models
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        svm = SVC(probability=True, random_state=42)
        nb = MultinomialNB()

        # Ensemble model
        self.ensemble_model = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("lr", lr), ("svm", svm), ("nb", nb)],
            voting="soft",
        )

        # Train ensemble
        self.ensemble_model.fit(X_train_combined, y_train)

        # Evaluate model
        print("Evaluating model...")
        y_pred = self.ensemble_model.predict(X_test_combined)
        y_pred_proba = self.ensemble_model.predict_proba(X_test_combined)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

        # Cross-validation
        cv_scores = cross_val_score(
            self.ensemble_model, X_train_combined, y_train, cv=5, scoring="roc_auc"
        )
        print(
            f"Cross-validation ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        self.is_trained = True

        # Save model
        self.save_model()

        return True

    def save_model(self):
        """Save the trained model and vectorizer"""
        os.makedirs("models", exist_ok=True)

        model_data = {
            "vectorizer": self.vectorizer,
            "ensemble_model": self.ensemble_model,
            "suspicious_patterns": self.suspicious_patterns,
            "is_trained": self.is_trained,
        }

        with open("models/advanced_clickbait_detector.pkl", "wb") as f:
            pickle.dump(model_data, f)

        print("Model saved to models/advanced_clickbait_detector.pkl")

    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            with open("models/advanced_clickbait_detector.pkl", "rb") as f:
                model_data = pickle.load(f)

            self.vectorizer = model_data["vectorizer"]
            self.ensemble_model = model_data["ensemble_model"]
            self.suspicious_patterns = model_data["suspicious_patterns"]
            self.is_trained = model_data["is_trained"]

            print(
                "Model loaded successfully from models/advanced_clickbait_detector.pkl"
            )
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def detect_clickbait_score(self, headline: str) -> float:
        """Calculate clickbait score using the trained ensemble model"""
        if not self.is_trained:
            if not self.load_model():
                print("No trained model available. Please train the model first.")
                return 0.0

        # Preprocess the headline
        if not headline.strip():
            return 0.0

        try:
            # Extract TF-IDF features
            tfidf_features = self.vectorizer.transform([headline])

            # Extract handcrafted features
            handcrafted_features = self.extract_handcrafted_features([headline])

            # Combine features
            from scipy.sparse import hstack, csr_matrix

            combined_features = hstack(
                [tfidf_features, csr_matrix(handcrafted_features)]
            )

            # Get prediction probability
            clickbait_probability = self.ensemble_model.predict_proba(
                combined_features
            )[0][1]

            return float(clickbait_probability)

        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.0

    def predict_clickbait(self, headline: str, threshold: float = 0.5) -> tuple:
        """Predict if headline is clickbait with confidence score"""
        score = self.detect_clickbait_score(headline)
        is_clickbait = score >= threshold
        confidence = score if is_clickbait else (1 - score)

        return is_clickbait, score, confidence


if __name__ == "__main__":
    intent = input("Do want to train? (y/n) ")

    if intent == "y":
        # Train the model
        print("Training advanced clickbait detector...")
        detector = AdvancedClickbaitDetector()
        success = detector.train()

        if success:
            # Test examples
            test_headlines = [
                "You Won't Believe What Happened Next!",
                "Scientists discover new treatment for diabetes",
                "SHOCKING: This Simple Trick Will Change Your Life Forever",
                "Economic growth shows steady improvement in Q3",
                "10 Amazing Facts That Will Blow Your Mind",
                "Research paper published on climate change impacts",
            ]

            print("\nTesting model on sample headlines:")
            print("-" * 50)

            for headline in test_headlines:
                is_clickbait, score, confidence = detector.predict_clickbait(headline)
                status = "CLICKBAIT" if is_clickbait else "NORMAL"
                print(f"{status} (Score: {score:.3f}, Confidence: {confidence:.3f})")
                print(f"  '{headline}'")
                print()
        else:
            print("Training failed. Please check your data files.")
    else:
        while True:
            test_detector = AdvancedClickbaitDetector()
            headline = input("Enter a headline to check clickbait score: ")
            is_clickbait, score, confidence = test_detector.predict_clickbait(headline)
            status = "CLICKBAIT" if is_clickbait else "NORMAL"
            print(f"{status} Score: {score:.3f}", "lower is better")
            print(f"  '{headline}'")
            print()
