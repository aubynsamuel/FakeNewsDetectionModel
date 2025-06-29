from pathlib import Path
import warnings
import time
from sklearn.model_selection import train_test_split

from .data_loader import load_data
from .preprocessing import parallel_preprocess
from .training import OptimizedFakeNewsDetector

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    print("=== OPTIMIZED FAKE NEWS DETECTION TRAINING PIPELINE ===")
    start_time = time.time()
    
    current_script_dir = Path(__file__).parent
    project_root_dir = current_script_dir.parent
    fake_csv_path = project_root_dir / 'data' / 'Fake.csv'
    true_csv_path = project_root_dir / 'data' / 'True.csv'
    
    texts, labels, _ = load_data(true_path=true_csv_path, fake_path=fake_csv_path)
    if texts is None:
        return

    print("\nStarting parallel text preprocessing...")
    processed_texts = parallel_preprocess(texts)

    # Filter out empty texts and corresponding labels
    valid_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
    processed_texts = [processed_texts[i] for i in valid_indices]
    labels = labels[valid_indices]

    print(f"After preprocessing: {len(processed_texts)} valid texts")

    use_bert = len(processed_texts) <= 15000
    detector = OptimizedFakeNewsDetector(use_distilbert=use_bert, max_bert_samples=15000)

    print(f"Using {'DistilBERT' if use_bert else 'TF-IDF'} for feature extraction...")
    print("Extracting features...")
    X, y = detector.prepare_features(processed_texts, labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    detector.train_models(X_train, y_train, X_test, y_test)

    detector.evaluate_best_model(X_test, y_test)

    detector.save_artifacts("optimized_v2")

    total_time = time.time() - start_time
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Best model: {detector.best_model_name}")
    print(f"Best F1-Score: {detector.models[detector.best_model_name]['metrics']['f1_score']:.4f}")

    print(f"\nModel Performance Comparison:")
    for name, result in detector.models.items():
        metrics = result['metrics']
        print(f"{name}:")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Training Time: {metrics['training_time']:.2f}s")


if __name__ == "__main__":
    main()