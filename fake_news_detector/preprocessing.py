import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

_PATTERNS = [
    (re.compile(r'\b[A-Z]+\s*\(Reuters\)\s*[-–—]?\s*', re.IGNORECASE), ''),
    (re.compile(r'\(Reuters\)', re.IGNORECASE), ''),
    (re.compile(r'Reuters', re.IGNORECASE), ''),
    (re.compile(r'\b(?:WASHINGTON|NEW YORK|LONDON|PARIS|BERLIN|TOKYO|MOSCOW|BEIJING|DELHI)\s*[-–—]?\s*', re.IGNORECASE), ''),
    (re.compile(r'\b(?:AP|CNN|BBC|Fox News|NBC|CBS|ABC News)\b', re.IGNORECASE), ''),
    (re.compile(r'\bBy\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', re.IGNORECASE), ''),
    (re.compile(r'\S+@\S+\.\S+'), ''),
    (re.compile(r'http[s]?://\S+'), ''),
    (re.compile(r'[^a-zA-Z\s]'), ' '),
    (re.compile(r'\s+'), ' ')
]

def remove_source_artifacts_fast(text):
    """Optimized version of source artifact removal"""
    if not isinstance(text, str) or len(text) < 10:
        return ""

    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)

    return text.strip().lower()

def _process_text_chunk(text_chunk):
    """Internal helper to process a chunk of texts in parallel"""
    return [remove_source_artifacts_fast(text) for text in text_chunk]

def parallel_preprocess(texts, n_jobs=None):
    """Parallel preprocessing of texts using multiprocessing"""
    if n_jobs is None:
        n_jobs = min(cpu_count(), 8)  # Limit to prevent memory issues

    # Split texts into chunks
    chunk_size = max(1, len(texts) // n_jobs)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    print(f"Processing {len(texts)} texts in {len(chunks)} chunks using {n_jobs} processes...")

    with Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(_process_text_chunk, chunks), total=len(chunks), desc="Preprocessing chunks"))

    # Flatten results
    processed_texts = []
    for chunk_result in results:
        processed_texts.extend(chunk_result)

    return processed_texts

if __name__ == '__main__':
    sample_texts = [
        "WASHINGTON (Reuters) - This is a sample text. By John Doe. Email: test@example.com",
        "Another text from AP. Check out https://example.com"
    ]
    preprocessed = parallel_preprocess(sample_texts, n_jobs=2)
    print("\nOriginal:", sample_texts)
    print("Processed:", preprocessed)