import tensorflow as tf
import numpy as np
import os
import tensorflow_hub as hub

# --- Configuration ---
OUTPUT_DIR = "models"
TFLITE_MODEL_PATH = os.path.join(OUTPUT_DIR, "student_model.tflite")
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"  # Still needed for embeddings

# SNLI labels mapping
SNLI_LABELS = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

class EntailmentAnalyzer:
  def __init__(self):
    interpreter, use_model = self._load_models()
    self.interpreter = interpreter
    self.embedder = use_model
      
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
              "Please ensure the TFLite conversion completed successfully and 'student_model.tflite' exists."
          )
          return None, None

      print("Loading Universal Sentence Encoder (USE) model...")
      try:
          use_model = hub.load(USE_MODEL_URL)
          print(f"USE model loaded from: {USE_MODEL_URL}")
      except Exception as e:
          print(f"Error loading USE model: {e}")
          print(
              "Please ensure you have an internet connection to download the USE model if not cached."
          )
          return None, None

      return interpreter, use_model


  # --- Preprocessing Function ---
  def _preprocess_input(self, premise: str, hypothesis: str) -> np.ndarray:
      """
      Generates enhanced embeddings for a single premise-hypothesis pair
      using the Universal Sentence Encoder, matching the training pipeline.

      Args:
          premise (str): The premise sentence.
          hypothesis (str): The hypothesis sentence.
          use_model: The loaded TensorFlow Hub Universal Sentence Encoder model.

      Returns:
          np.ndarray: A 1D numpy array of combined enhanced features, as float32.
      """
      # Explicitly place operations on CPU for consistent behavior
      with tf.device("/CPU:0"):
          # Convert to TensorFlow tensors
          premise_tensor = tf.constant([premise])
          hypothesis_tensor = tf.constant([hypothesis])

          # Generate embeddings
          premise_embeddings = self.embedder(premise_tensor)
          hypothesis_embeddings = self.embedder(hypothesis_tensor)

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
          # Convert to numpy and flatten to 1D array for single inference
          # Crucially, ensure the dtype is float32 for TFLite compatibility
          return enhanced_features.numpy().astype(np.float32).flatten()


  # --- Prediction Function for TFLite ---
  def predict_nli_tflite(self, 
      premise: str, hypothesis: str
  ) -> str:
      """
      Makes a natural language inference prediction using the TFLite model.

      Args:
          premise (str): The premise sentence.
          hypothesis (str): The hypothesis sentence.
          tflite_interpreter: The loaded TFLite model interpreter.
          use_model: The loaded TensorFlow Hub Universal Sentence Encoder model.

      Returns:
          str: The predicted NLI label (Entailment, Neutral, or Contradiction).
      """
      # Preprocess the input to get the feature vector
      input_features = self._preprocess_input(premise, hypothesis)

      # Reshape for model prediction (add batch dimension and ensure float32)
      input_for_tflite = np.expand_dims(input_features, axis=0)  # Shape (1, input_dim)

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
        print("\n--- Running Inference Examples (using TFLite model) ---")

        while True:
          premise = input("Enter a premise: ")
          hypothesis = input("Enter a hypothesis: ")

          print(f"Premise:    '{premise}'")
          print(f"Hypothesis: '{hypothesis}'")

          predicted_label = entailmentAnalyzer.predict_nli_tflite(
              premise, hypothesis
          )

          print(f"Predicted:  '{predicted_label}'")
