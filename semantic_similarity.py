import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob

model = SentenceTransformer("all-mpnet-base-v2")
model.eval()


def calculate_semantic_similarity(claim: str, sentence: str) -> float:
    """
    Calculates a weighted score representing how well a list of sentences supports a claim.
    Args:
        claim (str): The claim to be verified.
        sentences (str): Sentences to check against the claim.
    Returns:
        float: A weighted score between 0.0 and 1.0.
    """
    if not sentence:
        return 0.1

    with torch.no_grad():
        claim_embedding = model.encode(claim, show_progress_bar=False)
        sentence_embedding = model.encode(sentence, show_progress_bar=False)
        cosine_score = util.cos_sim(claim_embedding, sentence_embedding)

        claim_sentiment = TextBlob(claim).sentiment.polarity
        sentence_sentiment = TextBlob(sentence).sentiment.polarity

        similarity = cosine_score.item()

        if claim_sentiment * sentence_sentiment > 0:
            similarity *= 1.1
        elif claim_sentiment * sentence_sentiment < 0:
            similarity *= 0.9

        # print(f"Sentence: {sentence}\nSimilarity: {similarity:.2f}\n")
        final_score = max(0.0, min(1.0, similarity))
    return final_score


if __name__ == "__main__":
    while True:
        claim_to_verify = input("Enter claim to verify: ")
        evidence = input("Enter evidence sentences: ")

        final_score = calculate_semantic_similarity(claim_to_verify, evidence)

        print(f"The final weighted support score for the claim is: {final_score:.4f}")

        if final_score > 0.65:
            print("Interpretation: The claim is strongly supported by the evidence. ✅")
        elif final_score > 0.4:
            print(
                "Interpretation: The claim has moderate support from the evidence. 🤔"
            )
        else:
            print("Interpretation: The claim has weak support from the evidence. ❌")
