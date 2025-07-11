# ✅ FakeNewsDetector – Expected Behavior

This is the **main orchestrator** that coordinates all modules to evaluate a news headline's credibility.

## 🔍 Primary Responsibilities

1. **Accept a raw headline (string) as input**
2. **Run multiple analyses** on the headline using the following sub-components:
   - Clickbait Detection
   - Source Credibility Analysis
   - Claim Verification
   - Network Propagation Analysis
3. **Aggregate scores** from these modules using defined weights
4. **Output a structured dictionary** with:
   - Final verdict (`Credible`, `Likely True`, `Unclear`, `Doubtful`, `False`)
   - A numerical credibility score
   - Confidence level (`Very High`, `High`, etc.)
   - Breakdown of individual components

## 🧩 Dependencies (Expected Responsibilities)

### 1. `AdvancedClickbaitDetector`

**Purpose**: Assign a score based on how "clickbait-y" the headline is.

**Expected Method**:

```python
def detect_clickbait_score(headline: str) -> float:
    # Returns a float between 0.0 (not clickbait) and 1.0 (extremely clickbait)
```

### 2. `SourceCredibilityAnalyzer`

**Purpose**: Evaluate the trustworthiness of a source based on its domain name.

**Expected Method**:

```python
def analyze_domain_credibility(domain: str) -> float:
    # Returns a float between 0.0 (untrustworthy) and 1.0 (highly trustworthy)
```

### 3. `ClaimVerifier`

**Purpose**: Compare the input claim/headline against actual web search results and estimate factual agreement.

**Expected Method**:

```python
def verify_claim_against_sources(headline: str, urls: List[str]) -> Dict:
    # Returns a dictionary like:
    # {"score": float between 0 and 1}
```

### 4. `NetworkAnalyzer`

**Purpose**: Check how widely and diversely the story is propagated across different domains.

**Expected Method**:

```python
def analyze_propagation_pattern(urls: List[str]) -> Dict:
    # Returns a dictionary like:
    # {
    #     "score": float between 0 and 1,
    #     "domain_diversity": float (percentage of unique domains)
    # }
```

Detailed Classification Report:
precision recall f1-score support

Entailment 0.8185 0.8394 0.8288 3368
Neutral 0.7714 0.7537 0.7624 3219
Contradiction 0.8326 0.8295 0.8310 3237

accuracy 0.8080 9824

macro avg 0.8075 0.8075 0.8074 9824

weighted avg 0.8077 0.8080 0.8078 9824

## dependencies

tensorflow universal-sentence-encoder -> **./deploy/main/claim_verifier.py and ./deploy/utils/debertaflow_inference.py**

clickbait_predictor(custom from scratch) **./models/clickbait/** -> **"./deploy/main/predict_clickbait.py"**

nli_predictor_model(distilled from "cross-encoder/nli-deberta-v3-small") **"./models/"** -> **./deploy/utils/debertaflow_inference.py**
