import pickle
import os
import warnings

from fake_news_detector.feature_extraction import DistilBERTFeatureExtractor
from fake_news_detector.preprocessing import remove_source_artifacts_fast


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def predict_fake_news(text, model_dir="models/"):
    """
    Predicts whether a given text is fake or true news.

    Args:
        text (str): The news article content to predict.
        model_dir (str): Directory where the trained models and components are saved.

    Returns:
        dict: A dictionary containing the prediction ('Fake' or 'True'),
              confidence score, and probabilities for each class.
              Returns an error message if something goes wrong.
    """
    try:
        # Load Metadata to determine which feature extractor was used
        metadata_path = os.path.join(model_dir, "optimized_v2_metadata.pkl")
        if not os.path.exists(metadata_path):
            return {
                "error": f"Metadata file not found: {metadata_path}. Please run the training script first."
            }

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        use_distilbert = metadata.get("use_distilbert", True)
        best_model_name = (
            metadata.get("best_model_name", "XGBoost").lower().replace(" ", "_")
        )

        # Load the Best Trained Model
        model_path = os.path.join(
            model_dir, f"optimized_v2_best_model_{best_model_name}.pkl"
        )
        if not os.path.exists(model_path):
            return {
                "error": f"Best model file not found: {model_path}. Did you train the model successfully?"
            }

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded best model: {metadata.get('best_model_name')}")

        # Preprocess the input text
        processed_text = remove_source_artifacts_fast(text)

        if not processed_text:
            return {
                "prediction": "Uncertain",
                "confidence": 0.5,
                "message": "Input text was too short or cleaned to an empty string after preprocessing.",
            }

        # Extract Features
        features = None
        if use_distilbert:
            config_path = os.path.join(model_dir, "optimized_v2_distilbert_config.pkl")
            if not os.path.exists(config_path):
                return {
                    "error": f"DistilBERT config file not found: {config_path}. Ensure DistilBERT was used in training."
                }

            with open(config_path, "rb") as f:
                bert_config = pickle.load(f)

            # Pass parameters from config to ensure consistency
            extractor = DistilBERTFeatureExtractor(
                model_name=bert_config.get("model_name", "distilbert-base-uncased"),
                max_length=bert_config.get("max_length", 256),
                batch_size=1,  # For single prediction, batch_size=1 is fine
            )
            features = extractor.extract_features([processed_text])
        else:
            tfidf_path = os.path.join(model_dir, "optimized_v2_tfidf_vectorizer.pkl")
            if not os.path.exists(tfidf_path):
                return {
                    "error": f"TF-IDF vectorizer file not found: {tfidf_path}. Ensure TF-IDF was used in training."
                }

            with open(tfidf_path, "rb") as f:
                vectorizer = pickle.load(f)
            features = vectorizer.transform([processed_text])

        if features is None:
            return {"error": "Failed to extract features from the input text."}

        # Make Prediction
        prediction_label = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]

        predicted_class = "True" if prediction_label == 1 else "Fake"
        confidence = max(prediction_proba)

        return {
            "prediction": predicted_class,
            "confidence": float(confidence),
            "probabilities": {
                "fake": float(prediction_proba[0]),
                "true": float(prediction_proba[1]),
            },
            "processed_text_sample": (
                processed_text[:200] + "..."
                if len(processed_text) > 200
                else processed_text
            ),
        }

    except Exception as e:
        return {"error": f"An error occurred during prediction: {e}"}


if __name__ == "__main__":
    print("\n--- Fake News Detector Interactive Test ---")
    print("Type your news article content. Type 'exit' to quit.")
    print("-" * 40)

    while True:
        user_input = input("\nEnter news article content: \n> ")
        if user_input.lower() == "exit":
            break

        if not user_input.strip():
            print("Please enter some text.")
            continue

        print("\nPredicting...")
        result = predict_fake_news(user_input)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\n--- Prediction Results ---")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(
                f"Probabilities - Fake: {result['probabilities']['fake']:.4f}, True: {result['probabilities']['true']:.4f}"
            )
            print(f"Processed Text (Sample): '{result['processed_text_sample']}'")
            print("-" * 40)

    print("Exiting prediction script. Goodbye!")
