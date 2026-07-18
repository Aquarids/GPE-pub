import json
import math
from collections import defaultdict
from pathlib import Path

from gpe.labels import Label, parse_label
from gpe.metrics import label_score, token_efficiency


LABELS = [label.value for label in Label]


def compute_metrics_from_jsonl(input_path, output_path=None):
    input_paths = (
        [Path(path) for path in input_path]
        if isinstance(input_path, (list, tuple, set))
        else [Path(input_path)]
    )
    rows = [
        json.loads(line)
        for path in input_paths
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]
    done_rows = [row for row in rows if row.get("status") == "done"]
    output = {
        "metadata": {
            "input_paths": [str(path) for path in input_paths],
            "rows": len(rows),
            "completed_rows": len(done_rows),
        },
        "results": grouped_metrics(done_rows, include_category=False),
        "by_category": grouped_metrics(done_rows, include_category=True),
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def grouped_metrics(rows, include_category):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row.get("method"),
            row.get("attack_type") or "clean",
            float(row.get("poison_ratio", 0)),
            row.get("evidence_source", "dataset"),
        )
        if include_category:
            key += (row.get("category") or "unknown",)
        groups[key].append(row)

    output = []
    for key, items in sorted(groups.items()):
        method, attack_type, ratio, evidence_source = key[:4]
        result = {
            "method": method,
            "attack_type": attack_type,
            "poison_ratio": ratio,
            "evidence_source": evidence_source,
        }
        if include_category:
            result["category"] = key[4]
        result.update(summarize_results(items))
        output.append(result)
    return output


def summarize_results(rows):
    overall_pairs = []
    subclaim_pairs = []
    overall_score = 0.0
    subclaim_score = 0.0
    overall_usage = defaultdict(int)
    subclaim_usage = defaultdict(int)
    poisoned_count = 0
    evidence_count = 0

    for row in rows:
        gold = normalized_label(row.get("gold"))
        predicted = normalized_label((row.get("prediction") or {}).get("label"))
        if gold and predicted:
            overall_pairs.append((gold, predicted))
            overall_score += label_score(gold, predicted)
        add_usage(overall_usage, row.get("overall_usage"))
        poisoned_count += int(row.get("poisoned_evidence_count", 0) or 0)
        evidence_count += int(row.get("evidence_count", 0) or 0)

        for item in row.get("subclaims", []) or []:
            subclaim_gold = normalized_label(item.get("gold"))
            subclaim_predicted = normalized_label((item.get("prediction") or {}).get("label"))
            if subclaim_gold and subclaim_predicted:
                subclaim_pairs.append((subclaim_gold, subclaim_predicted))
                subclaim_score += label_score(subclaim_gold, subclaim_predicted)
            add_usage(subclaim_usage, item.get("usage"))

    return {
        "actual_poison_ratio": poisoned_count / evidence_count if evidence_count else 0.0,
        "overall": metric_block(overall_pairs, overall_score, overall_usage),
        "subclaim": metric_block(subclaim_pairs, subclaim_score, subclaim_usage),
    }


def normalized_label(value):
    label = parse_label(value, strict=False)
    return label.value if label else None


def add_usage(target, usage):
    for key in ("request_count", "prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] += int((usage or {}).get(key, 0) or 0)


def metric_block(pairs, score_sum, usage):
    total = len(pairs)
    correct = sum(gold == predicted for gold, predicted in pairs)
    efficiency = token_efficiency(usage["total_tokens"], correct, total)
    tcv = efficiency["tokens_per_correct"]
    return {
        "total": total,
        "correct": correct,
        "exact_accuracy": correct / total if total else None,
        "mean_label_score": score_sum / total if total else None,
        "classification": classification_metrics(pairs),
        "token_usage": dict(usage),
        "tcv": tcv if math.isfinite(tcv) else None,
        "correct_per_1k_tokens": efficiency["correct_per_1k_tokens"],
        "tokens_per_prediction": efficiency["tokens_per_prediction"],
    }


def classification_metrics(pairs):
    matrix = {
        gold: {predicted: 0 for predicted in LABELS}
        for gold in LABELS
    }
    for gold, predicted in pairs:
        matrix[gold][predicted] += 1

    per_label = {}
    total_support = len(pairs)
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[gold][label] for gold in LABELS if gold != label)
        fn = sum(matrix[label][predicted] for predicted in LABELS if predicted != label)
        support = sum(matrix[label].values())
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_precision = sum(item["precision"] for item in per_label.values()) / len(LABELS)
    macro_recall = sum(item["recall"] for item in per_label.values()) / len(LABELS)
    macro_f1 = sum(item["f1"] for item in per_label.values()) / len(LABELS)
    weighted_precision = safe_divide(
        sum(item["precision"] * item["support"] for item in per_label.values()),
        total_support,
    )
    weighted_recall = safe_divide(
        sum(item["recall"] * item["support"] for item in per_label.values()),
        total_support,
    )
    weighted_f1 = safe_divide(
        sum(item["f1"] * item["support"] for item in per_label.values()),
        total_support,
    )
    total_tp = sum(item["tp"] for item in per_label.values())
    total_fp = sum(item["fp"] for item in per_label.values())
    total_fn = sum(item["fn"] for item in per_label.values())
    micro_precision = safe_divide(total_tp, total_tp + total_fp)
    micro_recall = safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = safe_divide(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_label": per_label,
        "confusion_matrix": {
            "labels": LABELS,
            "values": [
                [matrix[gold][predicted] for predicted in LABELS]
                for gold in LABELS
            ],
        },
    }


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0
