import json
import math
import re
from collections import Counter
from pathlib import Path

from gpe.resources import GPE_DATA_PATH
from gpe.retrieval.retrieval_skill import rank_evidence_summaries
from gpe.retrieval.source_scope import matches_source_scope


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
        yield evidence


class EvidenceRetriever:
    def __init__(
        self,
        data_path=None,
        documents=None,
        evidence_types=None,
        attack_types=None,
    ):
        self.data_path = Path(data_path) if data_path is not None else GPE_DATA_PATH
        self.evidence_types = set(evidence_types) if evidence_types is not None else None
        self.attack_types = set(attack_types) if attack_types is not None else None
        self.documents = []
        self.document_frequency = Counter()
        self.scope_statistics = {}
        if documents is None:
            documents = self._load_documents()
        for evidence in documents:
            if self._accept_evidence(evidence):
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
                    if not self._accept_evidence(evidence):
                        continue
                    documents.append({**evidence, "claim_id": claim_id})
        return documents

    def _accept_evidence(self, evidence):
        if (
            self.evidence_types is not None
            and evidence.get("evidence_type") not in self.evidence_types
        ):
            return False
        if (
            self.attack_types is not None
            and evidence.get("evidence_type") == "poisoned"
            and evidence.get("attack_type") not in self.attack_types
        ):
            return False
        return True

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
        source_scope="web",
    ):
        """Retrieve evidence with BM25 or LLM summary-based reranking.

        Omit ``claim_id`` to search the full corpus; provide one to restrict
        candidates to a benchmark claim. LLM mode uses BM25 to recall
        ``candidate_k`` records, then ranks their titles, summaries, and
        keywords without sending full evidence contents to the LLM.
        """
        if mode == "bm25":
            return self._search_bm25(
                query, top_k, filter_benign, claim_id, source_scope
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
                source_scope=source_scope,
            )
        raise ValueError("mode must be 'bm25' or 'llm'")

    def _scoped_documents(self, source_scope, filter_benign):
        key = (source_scope, bool(filter_benign))
        if key not in self.scope_statistics:
            documents = [
                item
                for item in self.documents
                if (not filter_benign or item[0].get("evidence_type") == "benign")
                and matches_source_scope(item[0], source_scope)
            ]
            frequencies = Counter()
            lengths = []
            for _, counts, length in documents:
                frequencies.update(counts.keys())
                lengths.append(length)
            average_length = sum(lengths) / len(lengths) if lengths else 0.0
            self.scope_statistics[key] = (documents, frequencies, average_length)
        return self.scope_statistics[key]

    def count(self, source_scope="web", filter_benign=False):
        documents, _, _ = self._scoped_documents(source_scope, filter_benign)
        return len(documents)

    def _search_bm25(
        self,
        query,
        top_k=10,
        filter_benign=False,
        claim_id=None,
        source_scope="web",
    ):
        terms = _tokens(query)
        if not terms or top_k < 1:
            return []
        documents, document_frequency, average_length = self._scoped_documents(
            source_scope, filter_benign
        )
        total = len(documents)
        if not total or not average_length:
            return []
        results = []
        for evidence, counts, length in documents:
            if claim_id and evidence["claim_id"] != claim_id:
                continue
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(1 + (total - document_frequency[term] + 0.5)
                                             / (document_frequency[term] + 0.5))
                denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length / average_length)
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
        source_scope="web",
    ):
        candidates = self._search_bm25(
            query, candidate_k, filter_benign, claim_id, source_scope
        )
        if not candidates:
            return []
        ranking_candidates = []
        by_id = {}
        for index, item in enumerate(candidates, start=1):
            opaque_id = f"candidate-{index:04d}"
            ranking_item = dict(item)
            ranking_item["record_id"] = opaque_id
            ranking_candidates.append(ranking_item)
            by_id[opaque_id] = item
        ranked = rank_evidence_summaries(llm, query, ranking_candidates, top_k)
        selected = []
        seen = set()
        for item in ranked:
            if item in by_id and item not in seen:
                selected.append(by_id[item])
                seen.add(item)
        selected.extend(item for item in candidates if item not in selected)
        return selected[:top_k]
