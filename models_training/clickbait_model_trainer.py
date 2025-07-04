import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings

from deploy.main.predict_clickbait import predict_clickbait

from deploy.utils.clickbait_utils import (
    extract_enhanced_features,
    clickbait_indicators,
)

warnings.filterwarnings("ignore")


class ClickbaitDetector:
    def __init__(self):
        self.classifier = None
        self.tfidf_vectorizer = None
        self.is_trained = False

    def _load_and_prepare_data(self):
        """Load and prepare training data"""
        print("Loading datasets...")

        # Load existing data files
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(current_script_dir)
        clickbait_csv_path = os.path.join(
            project_root_dir, "data", "clickbait_data.csv"
        )
        non_clickbait_csv_path = os.path.join(
            project_root_dir, "data", "non_clickbait_data.csv"
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

    def _train(
        self,
        output_dir="./models/clickbait",
        epochs=10,  # Increased epochs for Logistic Regression
    ):
        """Train the optimized hybrid model"""
        print("Starting optimized training...")

        # Load data
        texts, labels = self._load_and_prepare_data()
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
            max_features=5000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            stop_words="english",
        )
        train_tfidf_features = self.tfidf_vectorizer.fit_transform(train_texts)
        val_tfidf_features = self.tfidf_vectorizer.transform(val_texts)
        print("TF-IDF vectorizer trained.")

        print("Extracting features...")
        train_handcrafted_features = extract_enhanced_features(train_texts)
        val_handcrafted_features = extract_enhanced_features(val_texts)
        print("Features extracted.")

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
            pickle.dump(clickbait_indicators, f)

        self.is_trained = True
        print("Model and vectorizer saved.")

        return True


# Usage example
if __name__ == "__main__":
    detector = ClickbaitDetector()
    success = detector._train(epochs=200)  # Use default epochs for LR

    if success:
        # Test the model
        test_headlines = [
            "You Won't Believe What Happened Next!",
            "Scientists discover new treatment for diabetes",
            "10 Amazing Secrets That Will Change Your Life Forever",
            "Economic growth shows steady improvement in Q3",
            "This Simple Trick Will Shock You - Doctors Hate It!",
            "Research published on climate change impacts",
        ]

        print("\nTesting hybrid model:")
        print("-" * 60)

        for headline in test_headlines:
            is_clickbait, score, confidence = predict_clickbait(headline)
            status = "CLICKBAIT" if is_clickbait else "NORMAL"
            print(f"{status} (Score: {score:.3f}, Confidence: {confidence:.3f})")
            print(f"  '{headline}'")
            print()
