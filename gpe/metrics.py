import math

from gpe.dataloader import ClaimLoader
from gpe.labels import Label, ORDINAL_LABELS, parse_label


class Evaluator:
    def __init__(self, claim_loader=None):
        self.claim_loader = claim_loader or ClaimLoader()

    def evaluate_claim(self, claim_id, label):
        gold = parse_label(self.claim_loader._record(claim_id).get("ground_truth"))
        prediction = parse_label(label)
        return {"claim_id": str(claim_id), "gold": gold.value, "prediction": prediction.value, "correct": gold == prediction, "score": label_score(gold, prediction)}

    def evaluate_subclaims(self, claim_id, predictions):
        gold_items = self.claim_loader.decompose(claim_id)
        gold_by_id = {str(item.get("id")): item for item in gold_items if item.get("id")}
        pred_by_id = {str(item.get("id")): item for item in (predictions or []) if item.get("id")}
        details = []
        correct = 0
        for subclaim_id, gold_item in gold_by_id.items():
            pred_item = pred_by_id.get(subclaim_id)
            gold_label = parse_label(gold_item.get("label"))
            pred_label = parse_label(pred_item.get("label")) if pred_item else None
            is_correct = pred_label == gold_label
            correct += int(is_correct)
            details.append({"id": subclaim_id, "subclaim": gold_item.get("subclaim"), "gold": gold_label.value, "prediction": pred_label.value if pred_label else None, "correct": is_correct})
        total = len(gold_by_id)
        return {"claim_id": str(claim_id), "total": total, "correct": correct, "accuracy": correct / total if total else None, "missing_ids": [item["id"] for item in details if item["prediction"] is None], "extra_ids": sorted(set(pred_by_id) - set(gold_by_id)), "details": details}


def label_score(gold, prediction):
    gold = parse_label(gold)
    prediction = parse_label(prediction)
    if gold == prediction:
        return 1.0
    if gold == Label.UNCERTAIN or prediction == Label.UNCERTAIN:
        return 0.0
    distance = abs(ORDINAL_LABELS.index(gold) - ORDINAL_LABELS.index(prediction))
    return max(0.0, 1.0 - distance / (len(ORDINAL_LABELS) - 1))


def normalize_token_usage(usage):
    usage = dict(usage or {})
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    return {"request_count": int(usage.get("request_count", 0) or 0), "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": int(usage.get("total_tokens", prompt + completion) or 0)}


def token_efficiency(total_tokens, correct, total):
    total_tokens = max(0, int(total_tokens or 0))
    correct = max(0, int(correct or 0))
    total = max(0, int(total or 0))
    return {"tokens_per_prediction": total_tokens / total if total else None, "tokens_per_correct": total_tokens / correct if correct else math.inf, "correct_per_1k_tokens": 1000.0 * correct / total_tokens if total_tokens else 0.0}
