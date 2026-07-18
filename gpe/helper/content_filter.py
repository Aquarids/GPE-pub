import re


BOILERPLATE_PATTERNS = (
    r"^advertisement$",
    r"^advertisements$",
    r"^sponsored content$",
    r"^subscribe$",
    r"^sign up$",
    r"^log in$",
    r"^privacy policy$",
    r"^terms of use$",
    r"^cookie policy$",
    r"^all rights reserved\.?$",
    r"^share this article$",
    r"^follow us$",
    r"^related articles$",
)

SECTION_NOISE_TITLES = {
    "references",
    "external links",
    "see also",
    "notes",
    "further reading",
    "bibliography",
}


def clean_content_blocks(blocks, min_chars=20):
    cleaned = []
    seen = set()
    for block in blocks:
        text = normalize_text(block)
        if not is_content_text(text, min_chars=min_chars):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def normalize_text(text):
    text = str(text or "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_content_text(text, min_chars=20):
    if not text:
        return False
    lowered = text.lower().strip("#=:- ")
    if lowered in SECTION_NOISE_TITLES:
        return False
    if len(text) < min_chars:
        return False
    for pattern in BOILERPLATE_PATTERNS:
        if re.match(pattern, lowered):
            return False
    return True


def is_noise_section_title(title):
    return normalize_text(title).lower() in SECTION_NOISE_TITLES
