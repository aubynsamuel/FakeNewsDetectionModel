import pickle
import numpy as np
from deploy.utils.clickbait_utils import extract_enhanced_features


class ClickbaitPredictor:
    def __init__(self, model_dir="./models/clickbait"):
        try:
            with open(f"{model_dir}/logistic_regression_model.pkl", "rb") as f:
                self.classifier = pickle.load(f)
            with open(f"{model_dir}/tfidf_vectorizer.pkl", "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open(f"{model_dir}/feature_info.pkl", "rb") as f:
                self.clickbait_indicators = pickle.load(f)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.classifier = None
            self.tfidf_vectorizer = None
            self.clickbait_indicators = None

    def predict(self, headline, threshold=0.5):
        if self.classifier is None or self.tfidf_vectorizer is None:
            raise RuntimeError("Model or vectorizer not loaded.")
        tfidf_features = self.tfidf_vectorizer.transform([headline])
        handcrafted_features = extract_enhanced_features([headline])
        combined_features = np.hstack((tfidf_features.toarray(), handcrafted_features))
        lr_probs = self.classifier.predict_proba(combined_features)[0]
        lr_score = lr_probs[1]
        is_clickbait = lr_score >= threshold
        confidence = lr_score if is_clickbait else (1 - lr_score)
        return is_clickbait, lr_score, confidence


if __name__ == "__main__":
    predictor = ClickbaitPredictor()
    while True:
        headline = input("Enter a headline to check clickbait score: ")
        is_clickbait, score, confidence = predictor.predict(headline)
        status = "CLICKBAIT" if is_clickbait else "NORMAL"
        print(f"{status} (Score: {score:.3f}, Confidence: {confidence:.3f})")
        print(f"  '{headline}'")
        print()
