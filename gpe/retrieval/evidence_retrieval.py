import json
import math
import re
from collections import Counter
from pathlib import Path

from gpe.resources import GPE_DATA_PATH
from gpe.retrieval.retrieval_skill import rank_evidence_summaries


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
    for item in environment.get("related_distractor") or []:
        evidence = dict(item)
        evidence.setdefault("evidence_type", "related_distractor")
        evidence.setdefault("attack_type", None)
        evidence["contents"] = []
        yield evidence


class EvidenceRetriever:
    def __init__(self, data_path=None, documents=None):
        self.data_path = Path(data_path) if data_path is not None else GPE_DATA_PATH
        self.documents = []
        self.document_frequency = Counter()
        if documents is None:
            documents = self._load_documents()
        for evidence in documents:
            self._add_document(evidence)
        self.average_length = (sum(length for _, _, length in self.documents) / len(self.documents)
                               if self.documents else 0.0)

    def _load_documents(self):
        documents = []
        with self.data_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                claim_id = record["claim_id"]
                for evidence in _evidence_items(record):
                    documents.append({**evidence, "claim_id": claim_id})
        return documents

    def _add_document(self, evidence):
        evidence = dict(evidence)
        metadata = evidence.get("retrieval") or {}
        contents = evidence.get("contents") or []
        if isinstance(contents, str):
            contents = [contents]
        text = " ".join([
            str(evidence.get("title") or ""),
            str(evidence.get("summary") or ""),
            " ".join(contents),
            " ".join(metadata.get("keywords") or []),
            str(metadata.get("seo_description") or ""),
        ])
        counts = Counter(_tokens(text))
        self.document_frequency.update(counts)
        claim_id = str(evidence.get("claim_id") or "")
        evidence_id = str(evidence.get("evidence_id") or len(self.documents))
        self.documents.append(({
            **evidence,
            "claim_id": claim_id,
            "record_id": f"{claim_id}:{evidence_id}",
        }, counts, sum(counts.values())))

    def search(
        self,
        query,
        top_k=10,
        filter_benign=False,
        claim_id=None,
        mode="bm25",
        llm=None,
        candidate_k=30,
        exclude_related_distractors=False,
    ):
        """Retrieve evidence with BM25 or LLM summary-based reranking.

        Omit ``claim_id`` to search the full corpus; provide one to restrict
        candidates to a benchmark claim. By default, the corpus includes
        entity-related distractors with empty ``contents``; set
        ``exclude_related_distractors=True`` to omit them for this retrieval.
        LLM mode uses BM25 to recall
        ``candidate_k`` records, then ranks their titles, summaries, and
        keywords without sending full evidence contents to the LLM.
        """
        if mode == "bm25":
            return self._search_bm25(
                query, top_k, filter_benign, claim_id, exclude_related_distractors
            )
        if mode == "llm":
            if llm is None:
                raise ValueError("llm is required when mode='llm'")
            return self.search_llm(
                query,
                llm,
                top_k=top_k,
                candidate_k=candidate_k,
                filter_benign=filter_benign,
                claim_id=claim_id,
                exclude_related_distractors=exclude_related_distractors,
            )
        raise ValueError("mode must be 'bm25' or 'llm'")

    def _search_bm25(
        self,
        query,
        top_k=10,
        filter_benign=False,
        claim_id=None,
        exclude_related_distractors=False,
    ):
        terms = _tokens(query)
        if not terms or top_k < 1:
            return []
        total = len(self.documents)
        results = []
        for evidence, counts, length in self.documents:
            if filter_benign and evidence.get("evidence_type") != "benign":
                continue
            if (
                exclude_related_distractors
                and evidence.get("evidence_type") == "related_distractor"
            ):
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

    def search_llm(
        self,
        query,
        llm,
        top_k=10,
        candidate_k=30,
        filter_benign=False,
        claim_id=None,
        exclude_related_distractors=False,
    ):
        candidates = self._search_bm25(
            query,
            candidate_k,
            filter_benign,
            claim_id,
            exclude_related_distractors,
        )
        if not candidates:
            return []
        ranked = rank_evidence_summaries(llm, query, candidates, top_k)
        by_id = {item["record_id"]: item for item in candidates}
        selected = []
        seen = set()
        for item in ranked:
            if item in by_id and item not in seen:
                selected.append(by_id[item])
                seen.add(item)
        selected.extend(item for item in candidates if item not in selected)
        return selected[:top_k]
