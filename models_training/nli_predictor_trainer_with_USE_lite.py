import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import torch
from sentence_transformers import CrossEncoder
import sentencepiece as spm
from typing import List, Tuple

OUTPUT_DIR = "models"
TFLITE_MODEL_PATH = os.path.join(OUTPUT_DIR, "nli_predictor_model.tflite")

USE_MODEL_PATH = "kaggle/input/universal-sentence-encoder/tensorflow1/lite/2"  # replace with actual path to USE lite

SNLI_LABELS = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

print("GPU Setup:")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# PyTorch device for teacher
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch teacher device: {torch_device}")


class UniversalSentenceEncoder:
    """
    A clean wrapper for Universal Sentence Encoder Lite with SentencePiece tokenization.
    This class is integrated from the user's provided code and modified to support batching.
    """

    def __init__(self, model_path: str):
        """
        Initialize the Universal Sentence Encoder.

        Args:
            model_path: Path to the Universal Sentence Encoder model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
        self._load_tokenizer()

    def _load_model(self) -> None:
        """Load the Universal Sentence Encoder model from TensorFlow Hub."""
        try:
            self.model = hub.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def _load_tokenizer(self) -> None:
        """Load the SentencePiece tokenizer from the model assets."""
        spm_path = self.model_path + "/assets/universal_encoder_8k_spm.model"

        try:
            self.tokenizer = spm.SentencePieceProcessor()
            with tf.io.gfile.GFile(spm_path, mode="rb") as f:
                self.tokenizer.LoadFromSerializedProto(f.read())
            print(f"SentencePiece tokenizer loaded from {spm_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SentencePiece tokenizer: {e}")

    def _tokenize_sentences(
        self, sentences: List[str]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Tokenize a list of sentences and convert to the separate components
        (indices, values, dense_shape) for the USE Lite model.

        Args:
            sentences: List of sentences to tokenize

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: indices, values, and dense_shape
            of the batched, tokenized sentences.
        """
        if not sentences:
            return (
                tf.constant([], dtype=tf.int64, shape=(0, 2)),
                tf.constant([], dtype=tf.int64),
                tf.constant([0, 0], dtype=tf.int64),
            )

        # Encode sentences to token IDs
        token_ids_list = [
            self.tokenizer.EncodeAsIds(sentence) for sentence in sentences
        ]

        # Calculate dimensions for sparse tensor for the batch
        max_length = max(len(ids) for ids in token_ids_list)
        batch_size = len(token_ids_list)
        dense_shape = tf.constant([batch_size, max_length], dtype=tf.int64)

        # Flatten token IDs and create indices for the sparse tensor
        values = []
        indices = []
        for row_idx, sentence_tokens in enumerate(token_ids_list):
            for col_idx, token in enumerate(sentence_tokens):
                values.append(token)
                indices.append([row_idx, col_idx])

        return (
            tf.constant(indices, dtype=tf.int64),
            tf.constant(values, dtype=tf.int64),
            dense_shape,
        )

    def encode_sentence(self, sentence: str) -> np.ndarray:
        """
        Generate embedding for a single sentence.

        Args:
            sentence: The sentence to encode

        Returns:
            NumPy array of the sentence embedding with shape (embedding_dim,)
        """
        if not sentence:
            raise ValueError("No sentence provided")

        # Use the batch encoding method for a single sentence
        return self.encode_sentences([sentence])[
            0
        ]  # Get the first (and only) embedding

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.

        Args:
            sentences: List of sentences to encode

        Returns:
            NumPy array of sentence embeddings with shape (num_sentences, embedding_dim)
        """
        if not sentences:
            return np.array([])  # Return empty array if no sentences

        # Tokenize the sentences into separate components
        indices, values, dense_shape = self._tokenize_sentences(sentences)

        # Generate embeddings using the model by passing the components as keyword arguments
        embeddings_dict = self.model.signatures["default"](
            indices=indices, values=values, dense_shape=dense_shape
        )

        # Extract embeddings from the output dictionary
        embeddings = embeddings_dict["default"].numpy()

        return embeddings


class DistillationLoss(tf.keras.losses.Loss):
    """Custom loss combining hard labels and soft teacher knowledge"""

    def __init__(self, temperature=4.0, alpha=0.3, name="distillation_loss"):
        super().__init__(name=name)
        self.temperature = temperature
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # y_true is a dictionary with 'hard_labels' and 'soft_logits'
        hard_labels = y_true["hard_labels"]
        soft_logits = y_true["soft_logits"]

        # Hard label loss
        hard_loss = tf.keras.losses.sparse_categorical_crossentropy(
            hard_labels, y_pred, from_logits=True
        )

        # Soft label loss (knowledge distillation)
        student_soft = tf.nn.softmax(y_pred / self.temperature, axis=1)
        teacher_soft = tf.nn.softmax(soft_logits / self.temperature, axis=1)

        # KL divergence loss
        # Add a small epsilon for numerical stability in log calculation
        epsilon = tf.keras.backend.epsilon()
        soft_loss = tf.reduce_sum(
            teacher_soft
            * tf.math.log((teacher_soft + epsilon) / (student_soft + epsilon)),
            axis=1,
        )
        soft_loss *= self.temperature**2

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss


class SparseCategoricalAccuracyWithSoftTarget(tf.keras.metrics.Metric):
    """Custom metric to calculate SparseCategoricalAccuracy from y_true['hard_labels']"""

    def __init__(self, name="accuracy", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true is the dictionary, extract the hard labels
        hard_labels = y_true["hard_labels"]
        self.accuracy_metric.update_state(hard_labels, y_pred, sample_weight)

    def result(self):
        return self.accuracy_metric.result()

    def reset_state(self):
        self.accuracy_metric.reset_state()


def load_snli_tfds(split, num_samples=None):
    """Load SNLI dataset using TensorFlow Datasets"""
    print(f"Loading SNLI split: '{split}' from TensorFlow Datasets...")
    try:
        ds = tfds.load("snli", split=split, as_supervised=False)
        if num_samples is not None:
            ds = ds.take(num_samples)
            print(f"Limiting to {num_samples} samples for '{split}' split.")
        return ds
    except Exception as e:
        print(f"Error loading SNLI from TensorFlow Datasets: {e}")
        return None


def tfds_to_pandas(dataset):
    """Convert TensorFlow dataset to pandas DataFrame"""
    data = {"premise": [], "hypothesis": [], "label": []}
    if dataset is None:
        return pd.DataFrame(data)

    for example in dataset:
        try:
            premise = example["premise"].numpy().decode("utf-8")
            hypothesis = example["hypothesis"].numpy().decode("utf-8")
            label = example["label"].numpy()
            if label != -1:  # Filter out unlabeled examples
                data["premise"].append(premise)
                data["hypothesis"].append(hypothesis)
                data["label"].append(label)
        except Exception as e:
            continue
    return pd.DataFrame(data)


def load_snli_data_for_training(output_dir="tf_distilled_nli_artifacts"):
    """Load SNLI dataset with large sample sizes, with caching for pandas DataFrames."""
    print("Loading SNLI dataset for teacher prediction and student training...")

    # Define cache paths for pandas DataFrames
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    train_df_cache_path = os.path.join(output_dir, "train_df.pkl")
    val_df_cache_path = os.path.join(output_dir, "val_df.pkl")
    test_df_cache_path = os.path.join(output_dir, "test_df.pkl")

    train_df, val_df, test_df = None, None, None

    # Try loading from cache
    if (
        os.path.exists(train_df_cache_path)
        and os.path.exists(val_df_cache_path)
        and os.path.exists(test_df_cache_path)
    ):
        print("Attempting to load SNLI DataFrames from cache...")
        try:
            train_df = pd.read_pickle(train_df_cache_path)
            val_df = pd.read_pickle(val_df_cache_path)
            test_df = pd.read_pickle(test_df_cache_path)
            print("Successfully loaded DataFrames from cache.")
        except Exception as e:
            print(
                f"Error loading cached DataFrames: {e}. Re-processing data from TFDS."
            )
            train_df, val_df, test_df = (
                None,
                None,
                None,
            )  # Reset to re-process if loading fails

    # If not loaded from cache (or if loading failed), process from TFDS
    if train_df is None:  # This condition checks if data was NOT loaded from cache
        print(
            "Loading data from TensorFlow Datasets and converting to pandas DataFrames..."
        )
        # You might want to reduce these for initial testing on limited RAM
        train_ds = load_snli_tfds("train", 500000)
        val_ds = load_snli_tfds("validation", 10000)
        test_ds = load_snli_tfds("test", 10000)

        train_df = tfds_to_pandas(train_ds)
        val_df = tfds_to_pandas(val_ds)
        test_df = tfds_to_pandas(test_ds)

        # Save to cache for future runs
        try:
            train_df.to_pickle(train_df_cache_path)
            val_df.to_pickle(val_df_cache_path)
            test_df.to_pickle(test_df_cache_path)
            print("Successfully cached DataFrames for future runs.")
        except Exception as e:
            print(f"Warning: Could not save DataFrames to cache: {e}")

    print(f"Training samples loaded: {len(train_df)}")
    print(f"Validation samples loaded: {len(val_df)}")
    print(f"Test samples loaded: {len(test_df)}")

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more loaded DataFrames are empty.")

    return train_df, val_df, test_df


def generate_teacher_soft_labels(
    teacher_model,
    premises,
    hypotheses,
    batch_size=512,
    save_path_logits=None,
    save_path_hard=None,
):
    """Generate soft labels using PyTorch teacher model, optionally saving/loading"""
    if (
        save_path_logits
        and os.path.exists(save_path_logits)
        and save_path_hard
        and os.path.exists(save_path_hard)
    ):
        print(
            f"Loading cached soft labels from {save_path_logits} and hard labels from {save_path_hard}..."
        )
        reordered_logits = np.load(save_path_logits)
        snli_hard_labels = np.load(save_path_hard)
        return reordered_logits, snli_hard_labels

    print("Generating soft labels from PyTorch teacher model...")
    all_soft_labels = []

    for i in tqdm(range(0, len(premises), batch_size), desc="Generating soft labels"):
        batch_premises = premises[i : i + batch_size]
        batch_hypotheses = hypotheses[i : i + batch_size]
        batch_pairs = [(p, h) for p, h in zip(batch_premises, batch_hypotheses)]

        # Get raw logits from teacher
        batch_soft_labels = teacher_model.predict(batch_pairs)
        all_soft_labels.extend(batch_soft_labels)

    teacher_logits = np.array(all_soft_labels)

    # Reorder teacher logits to match SNLI label order
    # CrossEncoder: [Contradiction, Entailment, Neutral] → SNLI: [Entailment, Neutral, Contradiction]
    reorder_indices = [1, 2, 0]  # [Teacher_col_1, Teacher_col_2, Teacher_col_0]
    reordered_logits = teacher_logits[:, reorder_indices]

    # Generate hard labels from teacher predictions
    teacher_hard_labels = np.argmax(teacher_logits, axis=1)
    # Map teacher indices to SNLI labels
    teacher_to_snli_mapping = {
        0: 2,
        1: 0,
        2: 1,
    }  # Contradiction→2, Entailment→0, Neutral→1
    snli_hard_labels = [teacher_to_snli_mapping[label] for label in teacher_hard_labels]

    if save_path_logits and save_path_hard:
        print(
            f"Saving generated soft labels to {save_path_logits} and hard labels to {save_path_hard}..."
        )
        np.save(save_path_logits, reordered_logits)
        np.save(save_path_hard, snli_hard_labels)

    return reordered_logits, snli_hard_labels


def generate_enhanced_embeddings_tf(
    premises, hypotheses, use_embedder_instance, batch_size=512, save_path=None
):
    """
    Generate enhanced embeddings using the custom UniversalSentenceEncoder instance in batches,
    optionally saving/loading.
    """
    if save_path and os.path.exists(save_path):
        print(f"Loading cached embeddings from {save_path}...")
        return np.load(save_path)

    print(
        f"Generating enhanced embeddings for {len(premises)} samples using custom USE Lite embedder in batches..."
    )

    all_enhanced_features = []

    for i in tqdm(
        range(0, len(premises), batch_size), desc="Generating USE Lite embeddings"
    ):
        batch_premises = premises[i : i + batch_size]
        batch_hypotheses = hypotheses[i : i + batch_size]

        # Use the custom embedder's encode_sentences method for batch processing
        premise_embeddings = use_embedder_instance.encode_sentences(batch_premises)
        hypothesis_embeddings = use_embedder_instance.encode_sentences(batch_hypotheses)

        # Create interaction features
        element_wise_product = tf.multiply(premise_embeddings, hypothesis_embeddings)
        element_wise_diff = tf.abs(
            tf.subtract(premise_embeddings, hypothesis_embeddings)
        )

        # Combine all features
        enhanced_features = tf.concat(
            [
                premise_embeddings,
                hypothesis_embeddings,
                element_wise_product,
                element_wise_diff,
            ],
            axis=1,
        )
        all_enhanced_features.append(
            enhanced_features.numpy()
        )  # Convert to numpy to append

    combined_enhanced_features = np.vstack(all_enhanced_features)

    if save_path:
        print(f"Saving generated embeddings to {save_path}...")
        np.save(save_path, combined_enhanced_features)

    print(f"Enhanced feature dimension: {combined_enhanced_features.shape[1]}")
    return combined_enhanced_features


def create_student_model(
    input_dim, hidden_dims=[1024, 512, 256], num_classes=3, dropout_rate=0.3
):
    """Create TensorFlow student model"""
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])

    # Add hidden layers
    for hidden_dim in hidden_dims:
        model.add(tf.keras.layers.Dense(hidden_dim, activation=None))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # Add output layer
    model.add(
        tf.keras.layers.Dense(num_classes, activation=None)
    )  # No activation for logits

    return model


def data_generator(X_path, hard_labels_path, soft_labels_path):
    X = np.load(X_path, mmap_mode="r")  # Use mmap_mode for large files
    hard_labels = np.load(hard_labels_path, mmap_mode="r")
    soft_labels = np.load(soft_labels_path, mmap_mode="r")

    for i in range(len(X)):
        yield X[i], {"hard_labels": hard_labels[i], "soft_logits": soft_labels[i]}


def create_distillation_dataset_from_paths(
    X_path, hard_labels_path, soft_labels_path, batch_size=512
):
    """
    Create a TensorFlow dataset for distillation training by loading from saved numpy files.
    This avoids loading all data into memory at once.
    """
    # Load one sample to infer shapes and dtypes
    # Using mmap_mode="r" for sample loading to avoid full load
    sample_X = np.load(X_path, mmap_mode="r")[0]
    sample_hard_label = np.load(hard_labels_path, mmap_mode="r")[0]
    sample_soft_label = np.load(soft_labels_path, mmap_mode="r")[0]

    output_types = (tf.float32, {"hard_labels": tf.int32, "soft_logits": tf.float32})
    output_shapes = (
        tf.TensorShape([sample_X.shape[0]]),  # X will be (input_dim,)
        {
            "hard_labels": tf.TensorShape([]),
            "soft_logits": tf.TensorShape([sample_soft_label.shape[0]]),
        },  # soft_logits will be (num_classes,)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X_path, hard_labels_path, soft_labels_path),
        output_types=output_types,
        output_shapes=output_shapes,
    )

    dataset = dataset.shuffle(
        buffer_size=1024
    )  # Shuffle after loading, before batching
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def train_tensorflow_student(
    X_train_path,
    hard_labels_train_path,
    soft_labels_train_path,
    X_val_path,
    hard_labels_val_path,
    soft_labels_val_path,
    input_dim,
    epochs=20,
    batch_size=512,
    learning_rate=0.001,
    patience=7,
    output_dir="tf_distilled_nli_artifacts",
):
    """Train TensorFlow student model with knowledge distillation using a custom training loop"""

    print("Training TensorFlow student model with custom training loop...")
    os.makedirs(output_dir, exist_ok=True)

    # Create student model
    student_model = create_student_model(input_dim)

    # Custom loss and optimizer
    distillation_loss_fn = DistillationLoss(temperature=4.0, alpha=0.3)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=0.01
    )

    # Create datasets using the new function
    train_dataset = create_distillation_dataset_from_paths(
        X_train_path, hard_labels_train_path, soft_labels_train_path, batch_size
    )
    val_dataset = create_distillation_dataset_from_paths(
        X_val_path, hard_labels_val_path, soft_labels_val_path, batch_size
    )

    # Metrics
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    train_accuracy_metric = SparseCategoricalAccuracyWithSoftTarget(
        name="train_accuracy"
    )
    val_accuracy_metric = SparseCategoricalAccuracyWithSoftTarget(name="val_accuracy")

    # Early Stopping and Model Checkpointing
    best_val_accuracy = -1
    epochs_without_improvement = 0
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    @tf.function
    def train_step(x, y_true):
        with tf.GradientTape() as tape:
            y_pred = student_model(x, training=True)
            loss = distillation_loss_fn(y_true, y_pred)

        gradients = tape.gradient(loss, student_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))

        train_loss_metric.update_state(loss)
        train_accuracy_metric.update_state(y_true, y_pred)

    @tf.function
    def val_step(x, y_true):
        y_pred = student_model(x, training=False)
        loss = distillation_loss_fn(y_true, y_pred)

        val_loss_metric.update_state(loss)
        val_accuracy_metric.update_state(y_true, y_pred)

    print("Starting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Reset metrics at the start of each epoch
        train_loss_metric.reset_state()
        val_loss_metric.reset_state()
        train_accuracy_metric.reset_state()
        val_accuracy_metric.reset_state()

        # Training loop
        for step, (x_batch_train, y_batch_train) in enumerate(
            tqdm(train_dataset, desc="Training")
        ):
            train_step(x_batch_train, y_batch_train)

        # Validation loop
        for step, (x_batch_val, y_batch_val) in enumerate(
            tqdm(val_dataset, desc="Validation")
        ):
            val_step(x_batch_val, y_batch_val)

        # Log metrics
        train_loss = train_loss_metric.result().numpy()
        val_loss = val_loss_metric.result().numpy()
        train_accuracy = train_accuracy_metric.result().numpy()
        val_accuracy = val_accuracy_metric.result().numpy()

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        # Early Stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            print(
                f"Validation accuracy improved. Saving best model to {os.path.join(output_dir, 'best_student_model.keras')}"
            )
            student_model.save(
                os.path.join(output_dir, "best_student_model.keras")
            )  # Save the entire model with .keras extension
        else:
            epochs_without_improvement += 1
            print(
                f"Validation accuracy did not improve for {epochs_without_improvement} epochs."
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Load best weights after training (if early stopping occurred)
    final_model_path = os.path.join(output_dir, "best_student_model.keras")
    if os.path.exists(final_model_path):
        print("Loading best weights for final evaluation...")
        student_model = tf.keras.models.load_model(final_model_path)
    else:
        print("No best model saved. Using the last trained model.")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.show()

    return student_model, history


def train_cross_framework_distilled_nli(
    teacher_model_name="cross-encoder/nli-deberta-v3-small",
    use_model_path=USE_MODEL_PATH,  # Use the local path variable
    output_dir="tf_distilled_nli_artifacts",
):
    """Main training function for cross-framework knowledge distillation"""

    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for cache files
    embeddings_dir = os.path.join(output_dir, "embeddings")
    soft_labels_dir = os.path.join(output_dir, "soft_labels")
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(soft_labels_dir, exist_ok=True)

    # Define paths for cached files
    train_embeddings_path = os.path.join(embeddings_dir, "train_embeddings.npy")
    val_embeddings_path = os.path.join(embeddings_dir, "val_embeddings.npy")
    test_embeddings_path = os.path.join(embeddings_dir, "test_embeddings.npy")

    train_soft_labels_path = os.path.join(soft_labels_dir, "train_soft_labels.npy")
    train_hard_labels_path = os.path.join(soft_labels_dir, "train_hard_labels.npy")
    val_soft_labels_path = os.path.join(soft_labels_dir, "val_soft_labels.npy")
    val_hard_labels_path = os.path.join(soft_labels_dir, "val_hard_labels.npy")

    print("\n=== Phase 1: Loading Data ===")
    try:
        train_df, val_df, test_df = load_snli_data_for_training(output_dir=output_dir)
    except ValueError as e:
        print(f"Error loading data: {e}")
        return

    # Extract data
    train_premises = train_df["premise"].tolist()
    train_hypotheses = train_df["hypothesis"].tolist()
    train_true_labels = train_df["label"].tolist()

    val_premises = val_df["premise"].tolist()
    val_hypotheses = val_df["hypothesis"].tolist()
    val_true_labels = val_df["label"].tolist()

    test_premises = test_df["premise"].tolist()
    test_hypotheses = test_df["hypothesis"].tolist()
    test_true_labels = test_df["label"].tolist()

    print(
        f"\n=== Phase 2: Generating Soft Labels with PyTorch Teacher ({teacher_model_name}) ==="
    )
    try:
        teacher_model = CrossEncoder(teacher_model_name, device=torch_device)
        print("PyTorch teacher model loaded successfully.")
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        return

    # Generate and save soft labels to disk
    train_soft_labels, train_hard_labels = generate_teacher_soft_labels(
        teacher_model,
        train_premises,
        train_hypotheses,
        save_path_logits=train_soft_labels_path,
        save_path_hard=train_hard_labels_path,
    )
    val_soft_labels, val_hard_labels = generate_teacher_soft_labels(
        teacher_model,
        val_premises,
        val_hypotheses,
        save_path_logits=val_soft_labels_path,
        save_path_hard=val_hard_labels_path,
    )

    print("Teacher soft labels generated and saved successfully.")

    # Clear large pandas DFs and soft/hard labels from memory after saving
    del train_df, val_df, test_df
    import gc  # Garbage Collection

    gc.collect()

    print(f"\n=== Phase 3: Loading TensorFlow USE Lite Model (Custom Class) ===")
    try:
        # Instantiate custom UniversalSentenceEncoder class
        use_embedder_instance = UniversalSentenceEncoder(model_path=use_model_path)
        print(
            "Custom UniversalSentenceEncoder (USE Lite) instance created successfully."
        )
    except Exception as e:
        print(f"Error loading USE Lite model using custom class: {e}")
        print(
            "Please ensure the USE Lite model path is correct and accessible, and all dependencies are met."
        )
        return

    print(f"\n=== Phase 4: Generating Enhanced Embeddings with Custom USE Lite ===")
    # These functions now return the path to the saved embeddings
    X_train = generate_enhanced_embeddings_tf(
        train_premises,
        train_hypotheses,
        use_embedder_instance,  # Pass the instance of custom class
        save_path=train_embeddings_path,
    )
    X_val = generate_enhanced_embeddings_tf(
        val_premises,
        val_hypotheses,
        use_embedder_instance,  # Pass the instance of custom class
        save_path=val_embeddings_path,
    )
    X_test = generate_enhanced_embeddings_tf(
        test_premises,
        test_hypotheses,
        use_embedder_instance,  # Pass the instance of custom class
        save_path=test_embeddings_path,
    )

    # Determine input_dim from the first loaded test embedding
    # X_test is now a numpy array (from generate_enhanced_embeddings_tf)
    input_dim = int(X_test.shape[1])

    # Clear premise/hypothesis data and generated X_train, X_val (not X_test as it's needed for eval)
    del train_premises, train_hypotheses, val_premises, val_hypotheses
    del X_train, X_val  # These are now just paths, the actual data is on disk
    gc.collect()

    print(f"\n=== Phase 5: Training TensorFlow Student Model ===")
    student_model, history = train_tensorflow_student(
        train_embeddings_path,  # Pass paths instead of loaded arrays
        train_hard_labels_path,
        train_soft_labels_path,
        val_embeddings_path,
        val_hard_labels_path,
        val_soft_labels_path,
        input_dim=input_dim,
        epochs=20,
        batch_size=512,
        learning_rate=0.001,
        patience=7,
        output_dir=output_dir,
    )

    # Save the trained student model after training is complete
    trained_model_path = os.path.join(output_dir, "trained_student_model.keras")
    print(f"\nSaving trained student model to {trained_model_path}")
    student_model.save(trained_model_path)

    print(f"\n=== Phase 6: Evaluating Student Model ===")
    # For evaluation, load test data if not already in memory
    X_test_eval = np.load(
        test_embeddings_path
    )  # Load the actual numpy array for prediction
    test_true_labels_arr = np.array(
        test_true_labels
    )  # Ensure this is a numpy array if it's not already

    # Predict on test set
    test_predictions = student_model.predict(X_test_eval)
    y_pred_student = np.argmax(test_predictions, axis=1)

    # Calculate accuracy
    student_accuracy = accuracy_score(test_true_labels_arr, y_pred_student)
    print(f"Student Model Accuracy on Test Set: {student_accuracy:.4f}")

    # Detailed evaluation
    print("\nDetailed Classification Report:")
    print(
        classification_report(
            test_true_labels_arr,
            y_pred_student,
            target_names=["Entailment", "Neutral", "Contradiction"],
            digits=4,
        )
    )

    print(f"\n=== Phase 7: Saving Models and Artifacts ===")

    # Save USE model reference
    use_model_info = {
        "model_path": use_model_path,  # Changed from url to path
        "embedding_dimension": int(input_dim) // 4,  # Before concatenation
        "enhanced_dimension": int(input_dim),
    }

    # Save model configuration
    model_config = {
        "input_dim": int(input_dim),
        "hidden_dims": [1024, 512, 256],
        "num_classes": 3,
        "dropout_rate": 0.3,
        "use_model_info": use_model_info,
    }

    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # Save accuracy results
    results = {
        "test_accuracy": float(student_accuracy),
        "training_epochs": len(history["loss"]),
        "best_val_accuracy": float(max(history["val_accuracy"])),
        "teacher_model": teacher_model_name,
        "use_model_path": use_model_path,
    }

    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Cross-Framework Knowledge Distillation Complete ===")
    print(f"Final Test Accuracy: {student_accuracy:.4f}")
    print("Files saved for TFLite deployment:")
    print(f"1. Student model: {trained_model_path}")  # Point to the newly saved model
    print(f"2. Model config: {os.path.join(output_dir, 'model_config.json')}")
    print(f"3. Training results: {os.path.join(output_dir, 'training_results.json')}")

    return (
        student_model,
        student_accuracy,
        trained_model_path,
    )  # Return the path to the saved model


def convert_to_tflite(model_path, output_dir="tf_distilled_nli_artifacts"):
    """Convert the trained model to TFLite format"""
    print("\n=== Converting to TFLite ===")

    # Load the saved model (no custom objects needed for inference model)
    model = tf.keras.models.load_model(model_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert
    tflite_model = converter.convert()

    # Save TFLite model
    tflite_path = os.path.join(output_dir, "nli_predictor_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved to: {tflite_path}")
    print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")

    return tflite_path


if __name__ == "__main__":
    # Train the cross-framework distilled model
    OUTPUT_DIR = "tf_distilled_nli_artifacts"

    model, accuracy, trained_model_path = train_cross_framework_distilled_nli(
        teacher_model_name="cross-encoder/nli-deberta-v3-small",
        use_model_path=USE_MODEL_PATH,  # Pass the local path
        output_dir=OUTPUT_DIR,
    )

    print(f"\nFinal model achieved {accuracy:.1%} accuracy on test set")

    # Convert to TFLite using the path to the newly saved model
    if trained_model_path is not None:
        tflite_path = convert_to_tflite(trained_model_path, OUTPUT_DIR)
        print(f"TFLite model ready for deployment: {tflite_path}")
