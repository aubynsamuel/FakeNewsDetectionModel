"""Why this file: This file is to serve as a guide to using the lite version of the
universal sentence encoder. This is because the official guides either uses old APIs
or just dont work. Download the model files from kaggle (~28mb) and replace module_url
with the extracted files locations
"""

import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
from scipy.spatial.distance import cosine

module_url = "path/to/universal-sentence-encoder-2/lite/version"
model = hub.load(module_url)

spm_path = module_url + "/assets/universal_encoder_8k_spm.model"
if spm_path is None:
    raise RuntimeError(
        "SentencePiece model file not found within the loaded TF Hub module's assets."
    )

print(f"Discovered SentencePiece model path: {spm_path}")

sp = spm.SentencePieceProcessor()
with tf.io.gfile.GFile(spm_path, mode="rb") as f:
    sp.LoadFromSerializedProto(f.read())
print(f"SentencePiece model loaded from {spm_path}.")


def process_to_IDs_in_sparse_format(tokenizer, sentences):
    """
    An utility method that processes sentences with the sentence piece processor
    'tokenizer' and returns the results in tf.SparseTensor-similar format:
    (values, indices, dense_shape)
    """
    ids = [tokenizer.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (
        tf.constant(values, dtype=tf.int64),
        tf.constant(indices, dtype=tf.int64),
        tf.constant(dense_shape, dtype=tf.int64),
    )


sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding.",
    "Dogs are great pets.",
    "Cats are also wonderful animals.",
    "This is a totally unrelated sentence.",
]

values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, sentences)

# The USE Lite model expects inputs through specific keys in a dictionary
embeddings_dict = model.signatures["default"](
    values=values, indices=indices, dense_shape=dense_shape
)

# The output embedding is typically under the 'default' key for the lite model
message_embeddings = embeddings_dict["default"].numpy()

print("\nEmbeddings:")
for i, (sentence, embedding) in enumerate(zip(sentences, message_embeddings)):
    print(f'Sentence: "{sentence}"')
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding snippet: {embedding[:5]}...")  # Print first 5 dimensions
    print("-" * 30)


def calculate_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)


print("\nSemantic Similarities (Cosine Similarity):")
# Compare "Dogs are great pets." with "Cats are also wonderful animals."
sim1 = calculate_cosine_similarity(message_embeddings[2], message_embeddings[3])
print(f'Similarity between "{sentences[2]}" and "{sentences[3]}": {sim1:.4f}')

# Compare "The quick brown fox jumps over the lazy dog." with "I am a sentence for which I would like to get its embedding."
sim2 = calculate_cosine_similarity(message_embeddings[0], message_embeddings[1])
print(f'Similarity between "{sentences[0]}" and "{sentences[1]}": {sim2:.4f}')

# Compare "Dogs are great pets." with "This is a totally unrelated sentence."
sim3 = calculate_cosine_similarity(message_embeddings[2], message_embeddings[4])
print(f'Similarity between "{sentences[2]}" and "{sentences[4]}": {sim3:.4f}')
