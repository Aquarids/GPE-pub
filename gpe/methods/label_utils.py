GOE_LABELS = {"true", "mostly_true", "half_true", "mostly_false", "false"}
COMPARE_LABELS = GOE_LABELS | {"unknown"}


def _normalize_label_text(value):
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_goe_label(value):
    label = _normalize_label_text(value)
    if label not in GOE_LABELS:
        raise ValueError(f"invalid GoE label: {value!r}")
    return label


def normalize_compare_label(value):
    label = _normalize_label_text(value)
    if label not in COMPARE_LABELS:
        raise ValueError(f"invalid compare label: {value!r}")
    return label


def normalize_three_way_label(value):
    label = normalize_goe_label(value)
    if label in {"true", "mostly_true"}:
        return "true"
    if label in {"false", "mostly_false"}:
        return "false"
    return "half_true"


def normalize_compare_three_way_label(value):
    label = normalize_compare_label(value)
    if label == "unknown":
        return "unknown"
    return normalize_three_way_label(label)
