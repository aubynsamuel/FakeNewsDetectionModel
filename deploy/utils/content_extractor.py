import re
import time
import requests
from bs4 import BeautifulSoup


def extract_content(
    url: str,
    content_cache: dict,
    cache_key: str,
    get_user_agent,
    timeout: int,
    cache_size: int,
) -> str:
    """Enhanced content extraction with proper character encoding handling (extracted utility)."""
    if cache_key in content_cache:
        return content_cache[cache_key]

    try:
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
        if response.encoding is None or response.encoding.lower() in [
            "iso-8859-1",
            "ascii",
        ]:
            response.encoding = "utf-8"

        try:
            html_content = response.text
        except UnicodeDecodeError:
            try:
                html_content = response.content.decode("utf-8", errors="ignore")
            except UnicodeDecodeError:
                html_content = response.content.decode("latin-1", errors="replace")

        # Clean problematic characters
        html_content = html_content.replace("\ufffd", " ")
        html_content = re.sub(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]", " ", html_content
        )

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove irrelevant content
        for element in soup(
            ["script", "style", "header", "footer", "nav", "aside", "form", "iframe"]
        ):
            element.decompose()

        # Extract content
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
                extracted_text = " ".join(
                    [elem.get_text(separator=" ", strip=True) for elem in elements]
                )
                break

        if not extracted_text:
            content_elements = soup.find_all(
                ["p", "div"], class_=lambda x: x is None or "ad" not in str(x).lower()
            )
            extracted_text = " ".join(
                [elem.get_text(separator=" ", strip=True) for elem in content_elements]
            )

        if not extracted_text:
            extracted_text = soup.get_text(separator=" ", strip=True)

        # Normalize and clean
        try:
            import unicodedata

            extracted_text = unicodedata.normalize("NFKD", extracted_text)
        except:
            pass

        content = re.sub(r"\s+", " ", extracted_text).strip()
        content = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", " ", content)
        content = content[:8000]

        # Cache result
        if len(content_cache) >= cache_size:
            oldest_key = next(iter(content_cache))
            del content_cache[oldest_key]

        content_cache[cache_key] = content
        return content

    except requests.exceptions.Timeout:
        return ""
    except requests.exceptions.RequestException:
        return ""
    except Exception:
        return ""
