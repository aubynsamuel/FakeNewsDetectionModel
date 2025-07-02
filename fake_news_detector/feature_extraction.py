import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DistilBERTFeatureExtractor:
    """Extracts features using a pre-trained DistilBERT model."""

    def __init__(
        self, model_name="distilbert-base-uncased", max_length=256, batch_size=32
    ):
        print(f"Initializing DistilBERT feature extractor with {model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

    def extract_features_batch(self, texts):
        """Extract features from a batch of texts."""
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            features = (
                outputs.last_hidden_state[:, 0, :].cpu().numpy()
            )  # Use [CLS] token

        return features

    def extract_features(self, texts):
        """Extract features from all texts with batching."""
        print(f"Extracting DistilBERT features from {len(texts)} texts...")
        all_features = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing batches"):
            batch_texts = texts[i : i + self.batch_size]
            batch_features = self.extract_features_batch(batch_texts)
            all_features.append(batch_features)

            if i % (self.batch_size * 10) == 0:  # Clear GPU memory periodically
                torch.cuda.empty_cache()
                gc.collect()

        return np.vstack(all_features)


class TFIDFVectorizerWrapper:
    """Wrapper for TF-IDF Vectorizer to ensure consistent API."""

    def __init__(
        self,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        sublinear_tf=True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)


if __name__ == "__main__":
    sample_texts = ["This is a test sentence.", "Another sentence for testing."]

    # Test DistilBERT
    bert_extractor = DistilBERTFeatureExtractor(batch_size=1)
    bert_features = bert_extractor.extract_features(sample_texts)
    print(f"DistilBERT features shape: {bert_features.shape}")

    # Test TF-IDF
    tfidf_extractor = TFIDFVectorizerWrapper()
    tfidf_features = tfidf_extractor.fit_transform(sample_texts)
    print(f"TF-IDF features shape: {tfidf_features.shape}")
