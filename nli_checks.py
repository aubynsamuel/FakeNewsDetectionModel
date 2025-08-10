import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()


def advanced_claim_verifier_native(claim: str, evidence: str) -> dict:
    """
    Verifies a claim against potentially long evidence using chunking and a powerful NLI model
    with the native transformers library.

    Args:
        claim (str): The claim to be verified (hypothesis).
        evidence (str): The evidence text, which can be long (premise).

    Returns:
        dict: A dictionary containing the final support score, the predicted label,
              and the most relevant evidence chunk.
    """
    if not evidence or not claim:
        return {
            "support_score": 0.5, # Neutral default
            "prediction": "Neutral",
            "relevant_chunk": "N/A"
        }

    # CHUNKING STRATEGY
    chunks = evidence.split('\n\n')
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    if not chunks:
        return {
            "support_score": 0.5,
            "prediction": "Neutral",
            "relevant_chunk": "N/A"
        }


    with torch.no_grad():
        inputs = tokenizer(chunks, [claim] * len(chunks), truncation=True, padding=True, return_tensors="pt").to(device)

        outputs = model(**inputs)

        # Convert logits to probabilities using softmax
        probabilities = torch.softmax(outputs.logits, dim=-1)
        
        # Convert to a NumPy array for easier handling
        scores = probabilities.cpu().numpy()
    
    entailment_scores = scores[:, 0]
    
    # Find the chunk with the highest entailment score
    best_chunk_idx = np.argmax(entailment_scores)
    
    # The final support score is the highest probability of entailment found
    final_support_score = entailment_scores[best_chunk_idx]
    
    # Determine the final label based on the highest probability for that best chunk
    final_prediction_idx = np.argmax(scores[best_chunk_idx])
    
    label_map = ["Supported", "Neutral", "Contradicted"]
    final_prediction_label = label_map[final_prediction_idx]
    
    most_relevant_chunk = chunks[best_chunk_idx]

    return {
        "support_score": float(final_support_score),
        "prediction": final_prediction_label,
        "relevant_chunk": most_relevant_chunk
    }


if __name__ == "__main__":
    claim_to_verify = "The company's new 'QuantumLeap' chip is expected to double processing speeds."

    long_evidence = """
    A press release today from Innovate Corp announced the 'QuantumLeap' processor. CEO Jane Doe stated, "We are thrilled to unveil this technology. Our internal benchmarks show that the QuantumLeap chip doubles the processing speed of our previous generation, a major milestone for the industry."

    The announcement was met with cautious optimism. Analyst John Smith from TechAdvisory noted, "While the claims are impressive, we've seen bold promises before. Real-world performance will be the true test." He also pointed out that the chip's power consumption remains a concern for mobile applications, which was not addressed in the release.

    The new processor will first be available in Innovate Corp's high-end desktop line, slated for a Q4 release. Broader availability has not yet been announced.
    """

    print(f"\nVERIFYING CLAIM: '{claim_to_verify}'\n")
    
    result = advanced_claim_verifier_native(claim_to_verify, long_evidence)

    print(f"Final Support Score: {result['support_score']:.4f}")
    print(f"Overall Assessment: {result['prediction']} ✅")
    print("-" * 30)
    print("Most Relevant Evidence Found:")
    print(f"-> \"{result['relevant_chunk']}\"")

    print("\n" + "="*50 + "\n")
    
    # Example for contradiction
    claim_contradicted = "Implementing a four-day workweek increases employee productivity"
    evidence_contradicted = "A trial of the four-day workweek led to employees completing fewer tasks overall, with managers noting a decline in project delivery speed and more frequent missed deadlines"
    
    print(f"VERIFYING CLAIM: '{claim_contradicted}'\n")
    result_contra = advanced_claim_verifier_native(claim_contradicted, evidence_contradicted)
    
    print(f"Final Support Score: {result_contra['support_score']:.4f}")
    print(f"Overall Assessment: {result_contra['prediction']} ❌")
    print("-" * 30)
    print("Most Relevant Evidence Found:")
    print(f"-> \"{result_contra['relevant_chunk']}\"")