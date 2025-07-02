from sentence_transformers import SentenceTransformer, util

# Load the model (fast and suitable)
model = SentenceTransformer("distilbert-base-uncased")

# Get embeddings
while True:
    sentence1 = input("Enter sentence 1: ")
    sentence2 = input("Enter sentence 2: ")
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2)

    # Apply a power transformation to make lower scores more discriminative
    power_factor = 2.0  # You can adjust this value (e.g., 1.5, 2.5, 3.0)
    transformed_similarity = similarity.item() ** power_factor

    print(f"Similarity Score: {transformed_similarity:.4f}")
