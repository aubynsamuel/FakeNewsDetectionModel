import re
import time
import requests
from bs4 import BeautifulSoup
from newspaper import Article


def extract_content(
    url: str,
    content_cache: dict,
    cache_key: str,
    get_user_agent,
    timeout: int,
    cache_size: int,
) -> str:
    """Enhanced content extraction with newspaper3k fallback to BeautifulSoup."""
    if cache_key in content_cache:
        return content_cache[cache_key]

    try:
        # Try newspaper3k first
        article = Article(url)
        article.download()
        article.parse()
        
        content = article.text
        
        # If newspaper3k didn't get good content, fallback to BeautifulSoup
        if not content or len(content.strip()) < 100:
            content = _fallback_extraction(url, get_user_agent, timeout)
        
        # Clean and normalize content
        content = _clean_content(content)
        content = content[:10000]  # Increased from 8000
        
        # Cache result
        if len(content_cache) >= cache_size:
            oldest_key = next(iter(content_cache))
            del content_cache[oldest_key]

        content_cache[cache_key] = content
        return content

    except Exception:
        # If newspaper3k fails, try BeautifulSoup fallback
        try:
            content = _fallback_extraction(url, get_user_agent, timeout)
            content = _clean_content(content)
            content = content[:10000]
            
            if len(content_cache) >= cache_size:
                oldest_key = next(iter(content_cache))
                del content_cache[oldest_key]

            content_cache[cache_key] = content
            return content
        except Exception:
            return ""


def _fallback_extraction(url: str, get_user_agent, timeout: int) -> str:
    """Fallback extraction using BeautifulSoup."""
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    time.sleep(0.5)

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    # Handle encoding
    if response.encoding is None or response.encoding.lower() in ["iso-8859-1", "ascii"]:
        response.encoding = "utf-8"

    try:
        html_content = response.text
    except UnicodeDecodeError:
        try:
            html_content = response.content.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            html_content = response.content.decode("latin-1", errors="replace")

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove irrelevant content
    for element in soup(["script", "style", "header", "footer", "nav", "aside", "form", "iframe"]):
        element.decompose()

    # Extract content using selectors
    content_selectors = [
        "article",
        "main",
        '[role="main"]',
        ".content",
        ".article-content",
        ".post-content",
        ".entry-content",
        ".article-body",
    ]

    extracted_text = ""
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            extracted_text = " ".join([elem.get_text(separator=" ", strip=True) for elem in elements])
            break

    if not extracted_text:
        content_elements = soup.find_all(["p", "div"], class_=lambda x: x is None or "ad" not in str(x).lower())
        extracted_text = " ".join([elem.get_text(separator=" ", strip=True) for elem in content_elements])

    if not extracted_text:
        extracted_text = soup.get_text(separator=" ", strip=True)

    return extracted_text


def _clean_content(content: str) -> str:
    """Clean and normalize extracted content."""
    # Clean problematic characters
    content = content.replace("\ufffd", " ")
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]", " ", content)
    
    # Normalize unicode if available
    try:
        import unicodedata
        content = unicodedata.normalize("NFKD", content)
    except:
        pass

    # Normalize whitespace and clean
    content = re.sub(r"\s+", " ", content).strip()
    content = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", " ", content)
    
    return content