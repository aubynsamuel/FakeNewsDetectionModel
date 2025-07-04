import pickle
import numpy as np

from deploy.utils.clickbait_utils import extract_enhanced_features


def load_model(model_dir="./models/clickbait"):
    """Load the trained model"""
    try:
        with open(f"{model_dir}/logistic_regression_model.pkl", "rb") as f:
            classifier = pickle.load(f)

        # Load TF-IDF vectorizer
        with open(f"{model_dir}/tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)

        # Load feature info
        with open(f"{model_dir}/feature_info.pkl", "rb") as f:
            clickbait_indicators = pickle.load(f)

        print("Model loaded successfully")
        return classifier, tfidf_vectorizer, clickbait_indicators

    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_clickbait(headline, threshold=0.5):
    """Predict clickbait with hybrid approach"""
    classifier, tfidf_vectorizer, clickbait_indicators = load_model()

    tfidf_features = tfidf_vectorizer.transform([headline])
    handcrafted_features = extract_enhanced_features([headline])
    # Combine features
    combined_features = np.hstack((tfidf_features.toarray(), handcrafted_features))
    # Logistic Regression prediction
    lr_probs = classifier.predict_proba(combined_features)[0]
    lr_score = lr_probs[1]  # Probability of being clickbait
    is_clickbait = lr_score >= threshold
    confidence = lr_score if is_clickbait else (1 - lr_score)
    return is_clickbait, lr_score, confidence


if __name__ == "__main__":
    while True:
        headline = input("Enter a headline to check clickbait score: ")
        is_clickbait, score, confidence = predict_clickbait(headline)
        status = "CLICKBAIT" if is_clickbait else "NORMAL"
        print(f"{status} (Score: {score:.3f}, Confidence: {confidence:.3f})")
        print(f"  '{headline}'")
        print()
