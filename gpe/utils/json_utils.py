import json
import re


def _remove_code_fence(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _clean_json_text(text):
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def _extract_candidates(text):
    candidates = []

    fenced = re.findall(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    candidates.extend(fenced)

    generic_fenced = re.findall(r"```\s*([\s\S]*?)\s*```", text)
    candidates.extend(generic_fenced)

    obj_matches = re.findall(r"({[\s\S]*})", text)
    candidates.extend(obj_matches)

    arr_matches = re.findall(r"(\[[\s\S]*\])", text)
    candidates.extend(arr_matches)

    candidates.append(text)
    return candidates


def extract_json(response):
    if isinstance(response, (dict, list)):
        return response

    text = str(response)
    last_error = None

    for raw in _extract_candidates(text):
        candidate = _clean_json_text(_remove_code_fence(raw))
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception as e:
            last_error = e

    if last_error is not None:
        raise ValueError(f"Failed to parse JSON: {last_error}")
    raise ValueError("No JSON found in response")


def to_json(data):
    def _default(obj):
        if isinstance(obj, set):
            return list(obj)
        return str(obj)

    return json.dumps(data, ensure_ascii=False, indent=2, default=_default)


def compress_json(original_prompt):
    def _compress(match):
        raw = match.group(1)
        try:
            obj = json.loads(raw)
            return "```json\n" + json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n```"
        except Exception:
            return match.group(0)

    compressed = re.sub(r"```json\n([\s\S]*?)\n```", _compress, original_prompt)
    compressed = re.sub(r"\n{3,}", "\n\n", compressed)
    compressed = re.sub(r"[ \t]{2,}", " ", compressed)
    return compressed.strip()
