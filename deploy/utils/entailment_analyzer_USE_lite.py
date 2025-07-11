import tensorflow as tf
import numpy as np
import os
import tensorflow_hub as hub
import sentencepiece as spm
from typing import List, Tuple

# --- Configuration ---
OUTPUT_DIR = "models"
TFLITE_MODEL_PATH = os.path.join(OUTPUT_DIR, "nli_predictor_model_use_lite.tflite")
# This path will now be passed to the UniversalSentenceEncoder class
USE_MODEL_PATH = "universal-sentence-encoder-tensorflow1-lite-v2"
SNLI_LABELS = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}


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


class EntailmentAnalyzer:
    def __init__(self):
        interpreter, use_embedder = self._load_models()
        self.interpreter = interpreter
        self.embedder = use_embedder

    def _load_models(self):
        """Loads the TFLite student model interpreter and the Universal Sentence Encoder model."""
        print("Loading TFLite student model interpreter...")
        try:
            interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
            interpreter.allocate_tensors()
            print(f"TFLite model loaded from: {TFLITE_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
            print(
                "Please ensure the TFLite conversion completed successfully and 'nli_predictor_model.tflite' exists."
            )
            return None, None

        print("Loading Universal Sentence Encoder (USE) model...")
        try:
            # Initialize custom UniversalSentenceEncoder class
            use_embedder = UniversalSentenceEncoder(model_path=USE_MODEL_PATH)
            print(f"USE embedder initialized with model from: {USE_MODEL_PATH}")
        except Exception as e:
            print(f"Error initializing USE embedder: {e}")
            print(
                "Please ensure the USE model path is correct and accessible, and all dependencies are met."
            )
            return None, None

        return interpreter, use_embedder

    # --- Preprocessing Function ---
    def _preprocess_input(self, premise: str, hypothesis: str) -> np.ndarray:
        """
        Generates enhanced embeddings for a single premise-hypothesis pair
        using the Universal Sentence Encoder, matching the training pipeline.

        Args:
            premise (str): The premise sentence.
            hypothesis (str): The hypothesis sentence.

        Returns:
            np.ndarray: A 1D numpy array of combined enhanced features, as float32.
        """
        # Explicitly place operations on CPU for consistent behavior
        with tf.device("/CPU:0"):
            # Generate embeddings using the custom embedder's encode_sentence method
            premise_embeddings = self.embedder.encode_sentence(premise)
            hypothesis_embeddings = self.embedder.encode_sentence(hypothesis)

            # Ensure embeddings are 2D (batch_size, embedding_dim) for tf.multiply, tf.subtract, tf.concat
            premise_embeddings = tf.expand_dims(premise_embeddings, axis=0)
            hypothesis_embeddings = tf.expand_dims(hypothesis_embeddings, axis=0)

            # Create interaction features
            element_wise_product = tf.multiply(
                premise_embeddings, hypothesis_embeddings
            )
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
            # Convert to numpy and flatten to 1D array for single inference
            # Crucially, ensure the dtype is float32 for TFLite compatibility
            return enhanced_features.numpy().astype(np.float32).flatten()

    # --- Prediction Function for TFLite ---
    def predict_nli_tflite(self, premise: str, hypothesis: str) -> str:
        """
        Makes a natural language inference prediction using the TFLite model.

        Args:
            premise (str): The premise sentence.
            hypothesis (str): The hypothesis sentence.

        Returns:
            str: The predicted NLI label (Entailment, Neutral, or Contradiction).
        """
        # Preprocess the input to get the feature vector
        input_features = self._preprocess_input(premise, hypothesis)

        # Reshape for model prediction (add batch dimension and ensure float32)
        input_for_tflite = np.expand_dims(
            input_features, axis=0
        )  # Shape (1, input_dim)

        # Get input and output tensors
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Set the input tensor
        self.interpreter.set_tensor(input_details[0]["index"], input_for_tflite)

        # Invoke inference
        self.interpreter.invoke()

        # Get the output tensor (logits)
        logits = self.interpreter.get_tensor(output_details[0]["index"])

        # Get the predicted class index (highest logit)
        predicted_class_idx = np.argmax(logits, axis=1)[0]

        # Map the index to the human-readable label
        predicted_label = SNLI_LABELS.get(predicted_class_idx, "Unknown")

        return predicted_label


# --- Main Execution ---
if __name__ == "__main__":
    entailmentAnalyzer = EntailmentAnalyzer()

    if entailmentAnalyzer.interpreter is None or entailmentAnalyzer.embedder is None:
        print("Failed to load models. Exiting.")
    else:
        # print("\n--- Running Inference Examples (using TFLite model) ---")

        # data = []
        # accurate_predictions = 0
        # # Ensure 'snli_1.0_dev.jsonl' is in the same directory or provide full path
        # try:
        #     with open("snli_1.0_dev.jsonl", "r", encoding="utf-8") as f:
        #         for line in f:
        #             data.append(json.loads(line))
        # except FileNotFoundError:
        #     print(
        #         "Error: 'snli_1.0_dev.jsonl' not found. Please place it in the same directory."
        #     )
        #     exit()  # Exit if the data file is not found

        # data = data[:1000]  # Limit to 100 samples for demonstration
        # for i, sample in enumerate(data):
        #     # Skip samples with a '-' label (unlabeled)
        #     if sample["gold_label"] == "-":
        #         continue

        #     premise = sample["sentence1"]
        #     hypothesis = sample["sentence2"]

        #     print(f"\n--- Sample {i+1} ---")
        #     print(f"Premise:    '{premise}'")
        #     print(f"Hypothesis: '{hypothesis}'")

        #     predicted_label = entailmentAnalyzer.predict_nli_tflite(premise, hypothesis)
        #     if predicted_label.lower() == sample["gold_label"].lower():
        #         accurate_predictions += 1

        #     print(f"Predicted:  {predicted_label}")
        #     print(f"Expected:   {sample['gold_label']}")

        # print(f"\n--- Evaluation Summary ---")
        # if len(data) > 0:
        #     print(f"Processed {len(data)} samples.")
        #     print(f"Accuracy: {(accurate_predictions/len(data))*100:.2f}%")
        # else:
        #     print("No samples processed from 'snli_1.0_dev.jsonl'.")

        print("\n--- Interactive Prediction ---")
        while True:
            premise = input("Enter a premise (or 'quit' to exit): ")
            if premise.lower() == "quit":
                break
            hypothesis = input("Enter a hypothesis: ")

            print(f"Premise:    '{premise}'")
            print(f"Hypothesis: '{hypothesis}'")

            predicted_label = entailmentAnalyzer.predict_nli_tflite(premise, hypothesis)

            print(f"Predicted:  '{predicted_label}'")
