import json
import math
import re
from collections import Counter
from pathlib import Path

from gpe.resources import GPE_DATA_PATH


def _tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).casefold())


def _evidence_items(record):
    environment = record.get("evidence_environment") or {}
    for item in environment.get("benign") or []:
        evidence = dict(item)
        evidence.setdefault("evidence_type", "benign")
        evidence.setdefault("attack_type", None)
        yield evidence
    for attack, items in (environment.get("poisoned") or {}).items():
        for item in items or []:
            evidence = dict(item)
            evidence.setdefault("evidence_type", "poisoned")
            evidence.setdefault("attack_type", attack)
            yield evidence


class EvidenceRetriever:
    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path is not None else GPE_DATA_PATH
        self.documents = []
        self.document_frequency = Counter()
        with self.data_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                claim_id = record["claim_id"]
                for evidence in _evidence_items(record):
                    metadata = evidence.get("retrieval") or {}
                    text = " ".join([
                        str(evidence.get("title") or ""),
                        str(evidence.get("summary") or ""),
                        " ".join(evidence.get("contents") or []),
                        " ".join(metadata.get("keywords") or []),
                        str(metadata.get("seo_description") or ""),
                    ])
                    counts = Counter(_tokens(text))
                    self.document_frequency.update(counts)
                    self.documents.append(({
                        **evidence,
                        "claim_id": claim_id,
                        "record_id": f"{claim_id}:{evidence['evidence_id']}",
                    }, counts, sum(counts.values())))
        self.average_length = (sum(length for _, _, length in self.documents) / len(self.documents)
                               if self.documents else 0.0)

    def search(self, query, top_k=10, filter_benign=False, claim_id=None):
        terms = _tokens(query)
        if not terms or top_k < 1:
            return []
        total = len(self.documents)
        results = []
        for evidence, counts, length in self.documents:
            if filter_benign and evidence.get("evidence_type") != "benign":
                continue
            if claim_id and evidence["claim_id"] != claim_id:
                continue
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(1 + (total - self.document_frequency[term] + 0.5)
                                             / (self.document_frequency[term] + 0.5))
                denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length / self.average_length)
                score += inverse_frequency * frequency * 2.2 / denominator
            if score:
                result = dict(evidence)
                result["score"] = score
                results.append(result)
        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

    def search_llm(self, query, llm, top_k=10, candidate_k=30, filter_benign=False, claim_id=None):
        candidates = self.search(query, candidate_k, filter_benign, claim_id)
        if not candidates:
            return []
        prompt = {
            "query": query,
            "task": "Rank evidence by relevance. Return JSON only: {\"record_ids\":[...]}",
            "evidence": [
                {"record_id": item["record_id"], "title": item.get("title"),
                 "summary": item.get("summary"),
                 "keywords": (item.get("retrieval") or {}).get("keywords", [])}
                for item in candidates
            ],
        }
        if callable(llm):
            response = llm(json.dumps(prompt, ensure_ascii=False))
        else:
            response = llm.dialogue([
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ], temperature=0)
        try:
            payload = json.loads(re.sub(r"^```(?:json)?|```$", "", str(response).strip()).strip())
            ranked = payload.get("record_ids") or []
        except (json.JSONDecodeError, AttributeError):
            ranked = []
        by_id = {item["record_id"]: item for item in candidates}
        selected = [by_id[item] for item in ranked if item in by_id]
        selected.extend(item for item in candidates if item not in selected)
        return selected[:top_k]
