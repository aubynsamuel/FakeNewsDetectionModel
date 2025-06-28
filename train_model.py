import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
import os

# Download NLTK data (run once) - Fixed exception handling
print("Checking and downloading required NLTK data...")
try:
    nltk.data.find('corpora/stopwords')
    print("Stopwords already downloaded")
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
    print("WordNet already downloaded")
except LookupError:
    print("Downloading wordnet...")
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
    print("Punkt tokenizer already downloaded")
except LookupError:
    print("Downloading punkt...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/omw-1.4')
    print("OMW already downloaded")
except LookupError:
    print("Downloading omw-1.4...")
    nltk.download('omw-1.4')
    
try:
    nltk.data.find('punkt_tab')
    print("punkt_tab already downloaded")
except LookupError:
    print("Downloading punkt_tab...")
    nltk.download('punkt_tab')
# --- 1. Data Loading and Preprocessing ---

def preprocess_text(text):
    """
    Cleans and preprocesses text data.
    """
    if not isinstance(text, str):
        return ""
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    # Tokenization and Stop words removal
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = [lemmatizer.lemmatize(word) for word in filtered_sentence]
    return " ".join(lemmatized_sentence)

print("\nLoading datasets...")
try:
    df_fake = pd.read_csv('Fake.csv')
    df_true = pd.read_csv('True.csv')
    print(f"Loaded {len(df_fake)} fake news articles and {len(df_true)} true news articles")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure 'Fake.csv' and 'True.csv' are in the same directory as this script.")
    exit()

# Add labels
df_fake['label'] = 0  # 0 for fake news
df_true['label'] = 1   # 1 for true news

# Combine datasets
df = pd.concat([df_fake, df_true]).sample(frac=1).reset_index(drop=True)
print(f"Combined dataset size: {len(df)} articles")

# Select relevant columns (e.g., 'title' and 'text')
# You might want to combine 'title' and 'text' for richer features
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')

print("Preprocessing text data...")
df['processed_content'] = df['content'].apply(preprocess_text)

# Remove empty processed content
df = df[df['processed_content'].str.strip() != ''].reset_index(drop=True)
print(f"Dataset size after preprocessing: {len(df)} articles")

# --- 2. Feature Extraction ---

print("Extracting features using TF-IDF...")
# Initialize TF-IDF Vectorizer
# max_features limits the number of features to prevent overly sparse matrices
# ngram_range can capture sequences of words (e.g., bigrams)
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000, 
    ngram_range=(1, 2),
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95  # Ignore terms that appear in more than 95% of documents
)

# Fit and transform the processed text data
X = tfidf_vectorizer.fit_transform(df['processed_content'])
y = df['label']

print(f"Feature matrix shape: {X.shape}")

# --- 3. Splitting Data ---

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# --- 4. Model Training (Gradient Boosting Classifier) ---

print("Training Gradient Boosting Classifier...")
# Initialize the Gradient Boosting Classifier
# You can tune these parameters for better performance
gb_classifier = GradientBoostingClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42,
    verbose=1  # Show training progress
)

# Train the model
gb_classifier.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Model Evaluation ---

print("Evaluating the model...")
# Make predictions on the test set
y_pred = gb_classifier.predict(X_test)
y_pred_proba = gb_classifier.predict_proba(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# Interpretation of Confusion Matrix:
# [[TN, FP],
#  [FN, TP]]
# TN (True Negatives): Correctly predicted fake news
# FP (False Positives): Incorrectly predicted true news (actually fake)
# FN (False Negatives): Incorrectly predicted fake news (actually true)
# TP (True Positives): Correctly predicted true news

# --- 6. Save the Model and Vectorizer ---

print("\nSaving the trained model and vectorizer...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the trained model
model_path = 'models/fake_news_detector.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(gb_classifier, f)
print(f"Model saved to: {model_path}")

# Save the TF-IDF vectorizer
vectorizer_path = 'models/tfidf_vectorizer.pkl'
with open(vectorizer_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"Vectorizer saved to: {vectorizer_path}")

# Save model metadata
metadata = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'feature_count': X.shape[1],
    'training_samples': X_train.shape[0],
    'test_samples': X_test.shape[0]
}

metadata_path = 'models/model_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"Model metadata saved to: {metadata_path}")

print(f"\nTraining and evaluation process complete!")
print(f"Files saved:")
print(f"  - {model_path}")
print(f"  - {vectorizer_path}")
print(f"  - {metadata_path}")

# --- 7. Test the saved model (optional verification) ---

print("\nVerifying saved model...")
try:
    # Load the saved model
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Load the saved vectorizer
    with open(vectorizer_path, 'rb') as f:
        loaded_vectorizer = pickle.load(f)
    
    # Test with a sample prediction
    test_accuracy = accuracy_score(y_test, loaded_model.predict(X_test))
    print(f"Loaded model accuracy: {test_accuracy:.4f}")
    print("Model verification successful!")
    
except Exception as e:
    print(f"Error verifying saved model: {e}")