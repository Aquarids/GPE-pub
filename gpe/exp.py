import json
import threading
from dataclasses import dataclass
from pathlib import Path

from gpe.dataloader import ClaimLoader, DatasetEvidenceLoader, read_jsonl
from gpe.metrics import Evaluator
from gpe.poison import PoisonCache
from gpe.retrieval.evidence_retrieval import EvidenceRetriever
from gpe.retrieval.source_scope import (
    matches_source_scope,
    project_poisoned_item,
    public_evidence_view,
    source_provenance,
)


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
    pool_top_k: int | None = None
    retrieval_source: str = "web"
    include_distractors: bool = False


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
    ):
        self.claims = claims
        self.evidence = evidence
        self.evaluator = evaluator
        self.llm = llm
        self.poison_cache = poison_cache
        self.global_pool = global_pool or GlobalEvidencePool(claims, evidence, poison_cache)

    def evaluate(self, case: EvaluationCase):
        claim = self.claims.get_claim(case.claim_id)
        before = self.llm.usage_snapshot()
        evidence_items = self.resolve_evidence(case.claim_id, case.evidence)
        visible_evidence = (
            [
                public_evidence_view(
                    item,
                    case.evidence.retrieval_source
                    if case.evidence.source == "global"
                    else "web",
                )
                for item in evidence_items
            ]
            if case.evidence.source != "search"
            else []
        )
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
            "retrieval_mode": case.evidence.retrieval_mode if case.evidence.source == "global" else None,
            "retrieval_top_k": case.evidence.top_k if case.evidence.source == "global" else None,
            "retrieval_candidate_k": case.evidence.retrieval_candidate_k if case.evidence.source == "global" else None,
            "pool_top_k": case.evidence.pool_top_k if case.evidence.source == "global" else None,
            "retrieval_source": case.evidence.retrieval_source if case.evidence.source == "global" else None,
            "include_distractors": case.evidence.include_distractors if case.evidence.source == "global" else None,
            "retrieval_pool_size": self.global_pool.size(case.evidence) if case.evidence.source == "global" else None,
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
        if request.source != "dataset":
            raise ValueError("evidence source must be dataset, global, or search")
        return self.evidence.get_evidence_list(
            claim_id,
            top_k=request.top_k,
            poison_ratio=request.poison_ratio,
            attack_type=request.attack_type,
            seed=request.seed,
            poison_cache=self.poison_cache,
            generate_missing_poison=request.generate_missing_poison,
        )

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


class GlobalEvidencePool:
    def __init__(self, claims, evidence, poison_cache, claim_ids=None):
        self.claims = claims
        self.evidence = evidence
        self.poison_cache = poison_cache
        self.claim_ids = {str(value) for value in claim_ids} if claim_ids is not None else None
        self.retrievers = {}
        self.retrieval_metadata = {}
        self.lock = threading.Lock()

    def search(self, query, request, llm):
        retriever = self._retriever(request)
        results = retriever.search(
            query,
            top_k=request.top_k or 3,
            filter_benign=request.filter_benign,
            mode=request.retrieval_mode,
            llm=llm,
            candidate_k=request.retrieval_candidate_k,
            source_scope=request.retrieval_source,
        )
        projected = []
        for item in results:
            item = project_poisoned_item(item, request.retrieval_source)
            item["source_provenance"] = source_provenance(
                item, request.retrieval_source
            )
            projected.append(item)
        return projected

    def size(self, request):
        return self._retriever(request).count(
            source_scope=request.retrieval_source,
            filter_benign=request.filter_benign,
        )

    def _retriever(self, request):
        pool_top_k = request.pool_top_k if request.pool_top_k is not None else request.top_k
        key = (
            float(request.poison_ratio),
            request.attack_type,
            request.seed,
            pool_top_k,
            request.retrieval_data_path,
            request.include_distractors,
            request.retrieval_source,
        )
        with self.lock:
            if key not in self.retrievers:
                self._load_retrieval_metadata(request.retrieval_data_path)
                documents = []
                for claim in self.claims.list_claims():
                    claim_id = claim["claim_id"]
                    if self.claim_ids is not None and str(claim_id) not in self.claim_ids:
                        continue
                    existing_identities = set()
                    for item in self.evidence.get_evidence_list(
                        claim_id,
                        top_k=pool_top_k,
                        poison_ratio=request.poison_ratio,
                        attack_type=request.attack_type,
                        seed=request.seed,
                        poison_cache=self.poison_cache,
                        generate_missing_poison=request.generate_missing_poison,
                    ):
                        item["claim_id"] = claim_id
                        metadata = self.retrieval_metadata.get((str(claim_id), str(item.get("evidence_id"))))
                        if metadata:
                            item["retrieval"] = metadata
                        documents.append(item)
                        existing_identities.add((
                            str(item.get("evidence_id") or ""),
                            str(item.get("url") or ""),
                        ))
                    if request.retrieval_source != "web":
                        for item in self.claims.benign_evidence(claim_id):
                            if not matches_source_scope(
                                item,
                                request.retrieval_source,
                                include_poisoned=False,
                            ):
                                continue
                            source_benign = dict(item)
                            source_benign.setdefault("evidence_type", "benign")
                            source_benign.setdefault("attack_type", None)
                            source_benign["claim_id"] = claim_id
                            identity = (
                                str(source_benign.get("evidence_id") or ""),
                                str(source_benign.get("url") or ""),
                            )
                            if identity in existing_identities:
                                continue
                            documents.append(source_benign)
                            existing_identities.add(identity)
                    if request.include_distractors:
                        record = self.claims._record(claim_id)
                        environment = record.get("evidence_environment") or {}
                        for item in environment.get("related_distractor") or []:
                            distractor = dict(item)
                            distractor.setdefault("evidence_type", "related_distractor")
                            distractor.setdefault("attack_type", None)
                            distractor["claim_id"] = claim_id
                            documents.append(distractor)
                            existing_identities.add((
                                str(distractor.get("evidence_id") or ""),
                                str(distractor.get("url") or ""),
                            ))
                self.retrievers[key] = EvidenceRetriever(documents=documents)
            return self.retrievers[key]

    def _load_retrieval_metadata(self, path):
        if not path or self.retrieval_metadata:
            return
        for record in read_jsonl(path):
            claim_id = str(record.get("claim_id") or "")
            environment = record.get("evidence_environment") or {}
            groups = [environment.get("benign") or []]
            groups.extend((environment.get("poisoned") or {}).values())
            for items in groups:
                for item in items or []:
                    metadata = item.get("retrieval")
                    if metadata:
                        self.retrieval_metadata[(claim_id, str(item.get("evidence_id")))] = metadata


class JsonlResultSink:
    def __init__(self, path, overwrite=False, rerun_missing_usage=False):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self.path.write_text("")
        self.completed = load_completed_keys(self.path, require_token_usage=rerun_missing_usage)

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
        case.evidence.retrieval_mode if case.evidence.source == "global" else None,
        case.evidence.top_k if case.evidence.source == "global" else None,
        case.evidence.retrieval_candidate_k if case.evidence.source == "global" else None,
        case.evidence.pool_top_k if case.evidence.source == "global" else None,
        case.evidence.retrieval_source if case.evidence.source == "global" else None,
        case.evidence.include_distractors if case.evidence.source == "global" else None,
    )


def row_key(row):
    return (
        row.get("method"),
        row.get("claim_id"),
        row.get("attack_type"),
        float(row.get("poison_ratio", 0)),
        row.get("evidence_source", "dataset"),
        row.get("retrieval_mode") if row.get("evidence_source") == "global" else None,
        row.get("retrieval_top_k") if row.get("evidence_source") == "global" else None,
        row.get("retrieval_candidate_k") if row.get("evidence_source") == "global" else None,
        row.get("pool_top_k") if row.get("evidence_source") == "global" else None,
        row.get("retrieval_source", "web") if row.get("evidence_source") == "global" else None,
        bool(row.get("include_distractors", False)) if row.get("evidence_source") == "global" else None,
    )


def load_completed_keys(path, require_token_usage=False):
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
            if require_token_usage and int((row.get("overall_usage") or {}).get("total_tokens", 0) or 0) <= 0:
                dirty = True
                continue
            valid_lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            completed.add(row_key(row))
    if dirty:
        path.write_text("".join(valid_lines), encoding="utf-8")
    return completed
