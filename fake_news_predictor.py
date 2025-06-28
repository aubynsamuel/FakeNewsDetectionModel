import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np

class FakeNewsDetector:
    def __init__(self, model_path='models/fake_news_detector.pkl', 
                 vectorizer_path='models/tfidf_vectorizer.pkl',
                 metadata_path='models/model_metadata.pkl'):
        """
        Initialize the Fake News Detector with pre-trained model and vectorizer.
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.metadata_path = metadata_path
        
        # Load model and vectorizer
        self.load_model()
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Downloading stopwords...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def load_model(self):
        """Load the trained model, vectorizer, and metadata."""
        try:
            # Load the trained model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load the TF-IDF vectorizer
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load metadata
            try:
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print("Model loaded successfully!")
                print(f"Model Accuracy: {self.metadata['accuracy']:.4f}")
                print(f"Model F1-Score: {self.metadata['f1_score']:.4f}")
            except FileNotFoundError:
                self.metadata = None
                print("Model loaded successfully (no metadata found)")
                
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Make sure you have trained the model first by running the training script.")
            raise
    
    def preprocess_text(self, text):
        """
        Preprocess text in the same way as during training.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove special characters, numbers, and extra spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenization and Stop words removal
        word_tokens = nltk.word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in self.stop_words]
        
        # Lemmatization
        lemmatized_sentence = [self.lemmatizer.lemmatize(word) for word in filtered_sentence]
        
        return " ".join(lemmatized_sentence)
    
    def predict(self, text):
        """
        Predict whether a news article is fake or real.
        
        Args:
            text (str): The news article text
            
        Returns:
            dict: Prediction results with label, probability, and confidence
        """
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        if not processed_text.strip():
            return {
                'prediction': 'unknown',
                'label': -1,
                'confidence': 0.0,
                'probabilities': {'fake': 0.5, 'real': 0.5}
            }
        
        # Transform text using the same vectorizer
        text_vector = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(max(probabilities) - 0.5) * 2
        
        result = {
            'prediction': 'real' if prediction == 1 else 'fake',
            'label': int(prediction),
            'confidence': confidence,
            'probabilities': {
                'fake': probabilities[0],
                'real': probabilities[1]
            }
        }
        
        return result
    
    def predict_batch(self, texts):
        """
        Predict multiple news articles at once.
        
        Args:
            texts (list): List of news article texts
            
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the detector
    try:
        detector = FakeNewsDetector()
        
        # Example news articles to test
        test_articles = [
            "Scientists discover new planet in our solar system with evidence of water and life forms.",
            "The stock market reached record highs today as investors remained optimistic about economic recovery.",
            "Breaking: Aliens have landed in Washington DC and are demanding to speak with world leaders immediately.",
            "Local school district announces new STEM program to enhance student learning in science and technology."
        ]
        
        print("\nTesting the Fake News Detector:")
        print("=" * 50)
        
        for i, article in enumerate(test_articles, 1):
            result = detector.predict(article)
            print(f"\nArticle {i}:")
            print(f"Text: {article[:100]}{'...' if len(article) > 100 else ''}")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Probabilities: Fake={result['probabilities']['fake']:.3f}, Real={result['probabilities']['real']:.3f}")
        
        # Interactive mode
        print("\n" + "=" * 50)
        print("Interactive Mode - Enter your own news articles to test:")
        print("(Type 'quit' to exit)")
        
        while True:
            user_input = input("\nEnter news article text: ").strip()
            if user_input.lower() == 'quit':
                break
            
            if user_input:
                result = detector.predict(user_input)
                print(f"\nPrediction: {result['prediction'].upper()}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Probabilities: Fake={result['probabilities']['fake']:.3f}, Real={result['probabilities']['real']:.3f}")
            else:
                print("Please enter some text.")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained the model first!")