import re


def _is_pdf_or_download_url(url: str) -> bool:
    """Check if URL points to a PDF or download file."""
    url_lower = url.lower()

    # Check for PDF in URL path
    if url_lower.endswith(".pdf"):
        return True

    # Check for PDF in URL path with query parameters
    if ".pdf?" in url_lower or ".pdf#" in url_lower:
        return True

    # Check for other document/download formats
    download_extensions = [
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".zip",
        ".rar",
        ".tar",
        ".gz",
        ".7z",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wmv",
        ".exe",
        ".msi",
        ".dmg",
        ".pkg",
        ".epub",
        ".mobi",
        ".djvu",
    ]

    for ext in download_extensions:
        if url_lower.endswith(ext) or f"{ext}?" in url_lower or f"{ext}#" in url_lower:
            return True

    # Check for common download URL patterns
    download_patterns = [
        r"/download/",
        r"/downloads/",
        r"/attachments/",
        r"/files/",
        r"/uploads/",
        r"/wp-content/uploads/",
        r"/content/uploads/",
        r"/assets/downloads/",
        r"/documents/",
        r"/pdfs/",
        r"\.pdf$",
        r"\.pdf\?",
        r"\.pdf#",
        r"attachment\.aspx",
        r"download\.aspx",
        r"getfile\.aspx",
        r"viewdocument\.aspx",
    ]

    return any(re.search(pattern, url_lower) for pattern in download_patterns)


def _is_corrupted_pdf_content(content: str) -> bool:
    """Detect if content appears to be corrupted PDF text."""
    if not content or len(content.strip()) < 10:
        return False

    # Common PDF corruption indicators
    pdf_corruption_patterns = [
        r"endstream\s+endobj",
        r"obj\s*<[^>]*>\s*stream",
        r"%PDF-\d+\.\d+",
        r"xref\s+\d+",
        r"trailer\s*<<",
        r"startxref",
        r"%%EOF",
        r"stream\s+H\s+[^\w\s]{10,}",  # Stream followed by garbled text
        r"[^\w\s]{20,}",  # Long sequences of non-word/space characters
        r"obj\s+<\s*>\s*stream",
        r"BT\s+/F\d+",  # PDF text object indicators
        r"ET\s+Q\s+q",  # PDF graphics state operators
    ]

    corruption_score = 0
    for pattern in pdf_corruption_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            corruption_score += 1

    # Check character distribution - PDFs often have weird character distributions
    if len(content) > 50:
        # Count non-printable or unusual characters
        unusual_chars = sum(
            1 for c in content if ord(c) > 127 or (ord(c) < 32 and c not in "\t\n\r ")
        )
        unusual_ratio = unusual_chars / len(content)

        if unusual_ratio > 0.3:  # More than 30% unusual characters
            corruption_score += 2

    # Check for excessive special characters in a row
    if re.search(r"[^\w\s]{15,}", content):
        corruption_score += 1

    # Check for PDF-specific garbled patterns
    if re.search(r"[A-Za-z0-9]{2,}\s+[^\w\s]{5,}\s+[A-Za-z0-9]{2,}", content):
        corruption_score += 1

    return corruption_score >= 2
