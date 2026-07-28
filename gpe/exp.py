import json
import threading
from dataclasses import dataclass
from pathlib import Path

from gpe.dataloader import ClaimLoader, DatasetEvidenceLoader, read_jsonl
from gpe.metrics import Evaluator
from gpe.poison import PoisonCache
from gpe.retrieval.evidence_retrieval import EvidenceRetriever


@dataclass(frozen=True)
class EvidenceRequest:
    source: str = "dataset"
    top_k: int | None = None
    poison_ratio: float = 0.0
    attack_type: str | None = None
    seed: int = 0
    generate_missing_poison: bool = True
    retrieval_mode: str = "bm25"
    retrieval_data_path: str | None = None
    retrieval_candidate_k: int = 30
    filter_benign: bool = False
    exclude_related_distractors: bool = False
    pool_top_k: int | None = None


@dataclass
class EvaluationCase:
    method: str
    detector: object
    claim_id: str
    evidence: EvidenceRequest
    include_subclaims: bool = True


class EvaluationPipeline:
    def __init__(
        self,
        claims: ClaimLoader,
        evidence: DatasetEvidenceLoader,
        evaluator: Evaluator,
        llm,
        poison_cache: PoisonCache | None = None,
        global_pool=None,
        claim_pool=None,
    ):
        self.claims = claims
        self.evidence = evidence
        self.evaluator = evaluator
        self.llm = llm
        self.poison_cache = poison_cache
        self.global_pool = global_pool or GlobalEvidencePool(claims, evidence, poison_cache)
        self.claim_pool = claim_pool or ClaimEvidencePool(claims, evidence, poison_cache)

    def evaluate(self, case: EvaluationCase):
        claim = self.claims.get_claim(case.claim_id)
        before = self.llm.usage_snapshot()
        evidence_items = self.resolve_evidence(case.claim_id, case.evidence)
        visible_evidence = evidence_items if case.evidence.source != "search" else []
        prediction, _ = run_prediction(
            case.detector,
            self.llm,
            claim["original_claim"],
            visible_evidence,
        )
        usage = self.llm.usage_delta(before)
        evaluation = self.evaluator.evaluate_claim(case.claim_id, prediction["label"])
        row = {
            "method": case.method,
            "claim_id": case.claim_id,
            "category": claim.get("category"),
            "attack_type": case.evidence.attack_type,
            "poison_ratio": case.evidence.poison_ratio,
            "evidence_source": case.evidence.source,
            "retrieval_mode": case.evidence.retrieval_mode if is_retrieval_source(case.evidence.source) else None,
            "retrieval_top_k": case.evidence.top_k if is_retrieval_source(case.evidence.source) else None,
            "retrieval_candidate_k": case.evidence.retrieval_candidate_k if is_retrieval_source(case.evidence.source) else None,
            "pool_top_k": effective_pool_top_k(case.evidence) if is_retrieval_source(case.evidence.source) else None,
            "retrieval_pool_size": self._pool_size(case.claim_id, case.evidence),
            "evidence_count": len(evidence_items),
            "poisoned_evidence_count": sum(bool(item.get("poisoned")) for item in evidence_items),
            "evidence_ids": [item.get("evidence_id") for item in evidence_items],
            "gold": claim.get("ground_truth"),
            "prediction": prediction,
            "correct": evaluation["correct"],
            "label_score": evaluation["score"],
            "overall_usage": usage,
            "status": "done",
        }
        if case.include_subclaims:
            row["subclaims"] = self.evaluate_subclaims(
                case.detector,
                case.claim_id,
                visible_evidence,
            )
        return row

    def resolve_evidence(self, claim_id, request: EvidenceRequest):
        if request.source == "search":
            return []
        if request.source == "global":
            claim = self.claims.get_claim(claim_id)
            return self.global_pool.search(claim["original_claim"], request, self.llm)
        if request.source == "local":
            claim = self.claims.get_claim(claim_id)
            return self.claim_pool.search(claim_id, claim["original_claim"], request, self.llm)
        if request.source != "dataset":
            raise ValueError("evidence source must be dataset, local, global, or search")
        return self.evidence.get_evidence_list(
            claim_id,
            top_k=request.top_k,
            poison_ratio=request.poison_ratio,
            attack_type=request.attack_type,
            seed=request.seed,
            poison_cache=self.poison_cache,
            generate_missing_poison=request.generate_missing_poison,
        )

    def _pool_size(self, claim_id, request):
        if request.source == "global":
            return self.global_pool.size(request)
        if request.source == "local":
            return self.claim_pool.size(claim_id, request)
        return None

    def evaluate_subclaims(self, detector, claim_id, evidence):
        results = []
        for subclaim in self.claims.decompose(claim_id):
            prediction, usage = run_prediction(
                detector,
                self.llm,
                subclaim["subclaim"],
                evidence,
            )
            results.append(
                {
                    "id": subclaim["id"],
                    "subclaim": subclaim["subclaim"],
                    "gold": subclaim["label"],
                    "prediction": prediction,
                    "usage": usage,
                }
            )
        return results


class ConditionedEvidencePool:
    def __init__(self, claims, evidence, poison_cache):
        self.claims = claims
        self.evidence = evidence
        self.poison_cache = poison_cache
        self.retrievers = {}
        self.retrieval_metadata = {}
        self.lock = threading.Lock()

    def _search(self, retriever, query, request, llm):
        return retriever.search(
            query,
            top_k=request.top_k or 3,
            filter_benign=request.filter_benign,
            mode=request.retrieval_mode,
            llm=llm,
            candidate_k=request.retrieval_candidate_k,
            exclude_related_distractors=request.exclude_related_distractors,
        )

    def _condition_key(self, request):
        return (
            float(request.poison_ratio),
            request.attack_type,
            request.seed,
            effective_pool_top_k(request),
            request.retrieval_data_path,
        )

    def _claim_documents(self, claim_id, request):
        metadata = self._load_retrieval_metadata(request.retrieval_data_path)
        documents = []
        for item in self.evidence.get_evidence_list(
            claim_id,
            top_k=effective_pool_top_k(request),
            poison_ratio=request.poison_ratio,
            attack_type=request.attack_type,
            seed=request.seed,
            poison_cache=self.poison_cache,
            generate_missing_poison=request.generate_missing_poison,
        ):
            item["claim_id"] = claim_id
            item_metadata = metadata.get((str(claim_id), str(item.get("evidence_id"))))
            if item_metadata:
                item["retrieval"] = item_metadata
            documents.append(item)
        for item in self.claims.related_distractors(claim_id):
            item["claim_id"] = claim_id
            documents.append(item)
        return documents

    def _benign_claim_documents(self, claim_id, request):
        metadata = self._load_retrieval_metadata(request.retrieval_data_path)
        documents = []
        for item in self.claims.benign_evidence(claim_id):
            item = dict(item)
            item["claim_id"] = claim_id
            item_metadata = metadata.get((str(claim_id), str(item.get("evidence_id"))))
            if item_metadata:
                item["retrieval"] = item_metadata
            documents.append(item)
        for item in self.claims.related_distractors(claim_id):
            item["claim_id"] = claim_id
            documents.append(item)
        return documents

    def _load_retrieval_metadata(self, path):
        if not path:
            return {}
        key = str(path)
        if key not in self.retrieval_metadata:
            metadata = {}
            for record in read_jsonl(path):
                claim_id = str(record.get("claim_id") or "")
                environment = record.get("evidence_environment") or {}
                groups = [environment.get("benign") or []]
                groups.extend((environment.get("poisoned") or {}).values())
                for items in groups:
                    for item in items or []:
                        values = item.get("retrieval")
                        if values:
                            metadata[(claim_id, str(item.get("evidence_id")))] = values
            self.retrieval_metadata[key] = metadata
        return self.retrieval_metadata[key]


class GlobalEvidencePool(ConditionedEvidencePool):
    def search(self, query, request, llm):
        return self._search(self._retriever(request), query, request, llm)

    def size(self, request):
        return len(self._retriever(request).documents)

    def _retriever(self, request):
        key = self._condition_key(request)
        with self.lock:
            if key not in self.retrievers:
                documents = []
                for claim in self.claims.list_claims():
                    documents.extend(self._claim_documents(claim["claim_id"], request))
                self.retrievers[key] = EvidenceRetriever(documents=documents)
            return self.retrievers[key]


class ClaimEvidencePool(ConditionedEvidencePool):
    def search(self, claim_id, query, request, llm):
        retrieved_benign = self._search(
            self._retriever(claim_id, request),
            query,
            request,
            llm,
        )
        return self.evidence.mix_selected_evidence(
            claim_id,
            retrieved_benign,
            poison_ratio=request.poison_ratio,
            attack_type=request.attack_type,
            seed=request.seed,
            poison_cache=self.poison_cache,
            generate_missing_poison=request.generate_missing_poison,
        )

    def size(self, claim_id, request):
        return len(self._retriever(claim_id, request).documents)

    def _retriever(self, claim_id, request):
        key = (str(claim_id), request.retrieval_data_path)
        with self.lock:
            if key not in self.retrievers:
                self.retrievers[key] = EvidenceRetriever(
                    documents=self._benign_claim_documents(claim_id, request)
                )
            return self.retrievers[key]


class JsonlResultSink:
    def __init__(self, path, overwrite=False):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self.path.write_text("")
        self.completed = load_completed_keys(self.path)

    def contains(self, case: EvaluationCase):
        with self._lock:
            return case_key(case) in self.completed

    def append(self, case: EvaluationCase, row):
        key = case_key(case)
        with self._lock:
            if key in self.completed:
                return False
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                file.flush()
            self.completed.add(key)
            return True


def build_pipeline(data_path, llm, poison_path=None, dynamic_poison_path=None):
    claims = ClaimLoader(data_path)
    evidence = DatasetEvidenceLoader(data_path, poison_path)
    return EvaluationPipeline(
        claims,
        evidence,
        Evaluator(claims),
        llm,
        PoisonCache(
            poison_path or data_path,
            dynamic_path=dynamic_poison_path,
            llm=llm,
            logger=getattr(llm, "logger", None),
        ),
    )


def run_prediction(detector, llm, text, evidence):
    before = llm.usage_snapshot()
    result = detector.detect(text, {"evidence": evidence})
    return {
        "label": result.get("verdict"),
        "confidence": result.get("confidence"),
        "explanation": result.get("explanation", ""),
    }, llm.usage_delta(before)


def case_key(case: EvaluationCase):
    return (
        case.method,
        case.claim_id,
        case.evidence.attack_type,
        float(case.evidence.poison_ratio),
        case.evidence.source,
        case.evidence.retrieval_mode if is_retrieval_source(case.evidence.source) else None,
        case.evidence.top_k if is_retrieval_source(case.evidence.source) else None,
        case.evidence.retrieval_candidate_k if is_retrieval_source(case.evidence.source) else None,
        effective_pool_top_k(case.evidence) if is_retrieval_source(case.evidence.source) else None,
    )


def row_key(row):
    return (
        row.get("method"),
        row.get("claim_id"),
        row.get("attack_type"),
        float(row.get("poison_ratio", 0)),
        row.get("evidence_source", "dataset"),
        row.get("retrieval_mode") if is_retrieval_source(row.get("evidence_source")) else None,
        row.get("retrieval_top_k") if is_retrieval_source(row.get("evidence_source")) else None,
        row.get("retrieval_candidate_k") if is_retrieval_source(row.get("evidence_source")) else None,
        row.get("pool_top_k") if is_retrieval_source(row.get("evidence_source")) else None,
    )


def effective_pool_top_k(request: EvidenceRequest):
    """Return the number of records contributed by each claim to a global pool."""
    return request.pool_top_k if request.pool_top_k is not None else request.top_k


def is_retrieval_source(source):
    return source in {"local", "global"}


def load_completed_keys(path):
    if not path.exists():
        return set()
    completed = set()
    valid_lines = []
    dirty = False
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                dirty = True
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring incomplete or invalid result at {path}:{line_number}")
                dirty = True
                continue
            if row.get("status") != "done":
                dirty = True
                continue
            valid_lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            completed.add(row_key(row))
    if dirty:
        path.write_text("".join(valid_lines), encoding="utf-8")
    return completed
