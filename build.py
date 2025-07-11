import gc

# import tensorflow_hub as hub
import nltk

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

gc.collect()

# USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4" # we are now using USE lite

# use_model = hub.load(USE_MODEL_URL)
