import time
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from .feature_extraction import DistilBERTFeatureExtractor, TFIDFVectorizerWrapper
from .models import get_available_models

class OptimizedFakeNewsDetector:
    """
    Manages feature extraction, model training, evaluation, and saving for fake news detection.
    """
    def __init__(self, use_distilbert=True, max_bert_samples=10000):
        self.use_distilbert = use_distilbert
        self.max_bert_samples = max_bert_samples
        self.feature_extractor = None # Initialized later based on use_distilbert
        self.tfidf_vectorizer = None # Initialized if TF-IDF is used
        self.models = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_features(self, texts, labels=None):
        """Prepare features using DistilBERT or TF-IDF."""
        if self.use_distilbert:
            if len(texts) > self.max_bert_samples:
                print(f"Sampling {self.max_bert_samples} texts for DistilBERT (dataset too large)")
                indices = np.random.choice(len(texts), self.max_bert_samples, replace=False)
                texts = [texts[i] for i in indices]
                if labels is not None:
                    labels = labels[indices]

            if self.feature_extractor is None:
                self.feature_extractor = DistilBERTFeatureExtractor(batch_size=16)
            features = self.feature_extractor.extract_features(texts)
            return features, labels if labels is not None else None
        else:
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TFIDFVectorizerWrapper()
                features = self.tfidf_vectorizer.fit_transform(texts)
            else:
                features = self.tfidf_vectorizer.transform(texts)

            return features, labels if labels is not None else None

    def train_models(self, X_train, y_train, X_test, y_test):
        """Train optimized models and evaluate them."""
        print("\n=== TRAINING OPTIMIZED MODELS ===")
        models = get_available_models()

        best_f1 = 0
        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()

            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"{name} Results (Training time: {training_time:.2f}s):")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")

            results[name] = {
                'model': model,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'training_time': training_time
                }
            }

            if f1 > best_f1:
                best_f1 = f1
                self.best_model = model
                self.best_model_name = name

        self.models = results
        print(f"\nBest model: {self.best_model_name} (F1: {best_f1:.4f})")
        return results

    def evaluate_best_model(self, X_test, y_test):
        """Performs a detailed evaluation of the best performing model."""
        if self.best_model is None:
            print("No model has been trained yet. Call train_models first.")
            return

        print(f"\n=== DETAILED EVALUATION: {self.best_model_name} ===")
        y_pred_best = self.best_model.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_best, target_names=['Fake', 'True']))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_best))

    def save_artifacts(self, prefix="optimized"):
        """Save trained models and feature extractors."""
        os.makedirs('models', exist_ok=True)

        # Save best model
        best_model_path = f'models/{prefix}_best_model_{self.best_model_name.lower().replace(" ", "_")}.pkl'
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Best model saved: {best_model_path}")

        # Save feature extractor (or its config)
        if self.use_distilbert and self.feature_extractor:
            extractor_info = {
                'model_name': 'distilbert-base-uncased',
                'max_length': self.feature_extractor.max_length,
                'batch_size': self.feature_extractor.batch_size
            }
            with open(f'models/{prefix}_distilbert_config.pkl', 'wb') as f:
                pickle.dump(extractor_info, f)
            print(f"DistilBERT config saved: models/{prefix}_distilbert_config.pkl")
        elif self.tfidf_vectorizer:
            tfidf_path = f'models/{prefix}_tfidf_vectorizer.pkl'
            with open(tfidf_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            print(f"TF-IDF vectorizer saved: {tfidf_path}")

        # Save metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'use_distilbert': self.use_distilbert,
            'model_results': {name: result['metrics'] for name, result in self.models.items()},
            'feature_type': 'DistilBERT' if self.use_distilbert else 'TF-IDF'
        }

        with open(f'models/{prefix}_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved: models/{prefix}_metadata.pkl")

if __name__ == '__main__':
    print("This is a training module. Run main.py for full pipeline.")