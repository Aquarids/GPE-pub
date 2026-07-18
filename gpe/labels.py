import re
from enum import Enum


class Label(Enum):
    TRUE = "true"
    MOSTLY_TRUE = "mostly_true"
    HALF_TRUE = "half_true"
    MOSTLY_FALSE = "mostly_false"
    FALSE = "false"
    UNCERTAIN = "uncertain"


LABEL_ALIASES = {label.value: label for label in Label}

ORDINAL_LABELS = [
    Label.FALSE,
    Label.MOSTLY_FALSE,
    Label.HALF_TRUE,
    Label.MOSTLY_TRUE,
    Label.TRUE,
]


def parse_label(value, strict=True):
    if isinstance(value, Label):
        return value
    label = LABEL_ALIASES.get(normalize_label_text(value))
    if label is not None:
        return label
    if strict:
        raise ValueError(f"unknown label: {value}")
    return None


def normalize_label_text(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")
