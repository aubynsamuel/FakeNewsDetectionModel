import gc
import tensorflow_hub as hub
import nltk

gc.collect()
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

use_model = hub.load(USE_MODEL_URL)
