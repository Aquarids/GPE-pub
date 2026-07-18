import json
import threading
from dataclasses import dataclass
from pathlib import Path

from gpe.dataloader import ClaimLoader, DatasetEvidenceLoader
from gpe.metrics import Evaluator
from gpe.poison import PoisonCache


@dataclass(frozen=True)
class EvidenceRequest:
    source: str = "dataset"
    top_k: int | None = None
    poison_ratio: float = 0.0
    attack_type: str | None = None
    seed: int = 0
    generate_missing_poison: bool = True


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
    ):
        self.claims = claims
        self.evidence = evidence
        self.evaluator = evaluator
        self.llm = llm
        self.poison_cache = poison_cache

    def evaluate(self, case: EvaluationCase):
        claim = self.claims.get_claim(case.claim_id)
        evidence_items = self.resolve_evidence(case.claim_id, case.evidence)
        visible_evidence = evidence_items if case.evidence.source == "dataset" else []
        prediction, usage = run_prediction(
            case.detector,
            self.llm,
            claim["original_claim"],
            visible_evidence,
        )
        evaluation = self.evaluator.evaluate_claim(case.claim_id, prediction["label"])
        row = {
            "method": case.method,
            "claim_id": case.claim_id,
            "category": claim.get("category"),
            "attack_type": case.evidence.attack_type,
            "poison_ratio": case.evidence.poison_ratio,
            "evidence_source": case.evidence.source,
            "evidence_count": len(evidence_items) if case.evidence.source == "dataset" else None,
            "poisoned_evidence_count": (
                sum(bool(item.get("poisoned")) for item in evidence_items)
                if case.evidence.source == "dataset"
                else None
            ),
            "evidence_ids": (
                [item.get("evidence_id") for item in evidence_items]
                if case.evidence.source == "dataset"
                else []
            ),
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
        if request.source != "dataset":
            raise ValueError("evidence source must be dataset or search")
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
    )


def row_key(row):
    return (
        row.get("method"),
        row.get("claim_id"),
        row.get("attack_type"),
        float(row.get("poison_ratio", 0)),
        row.get("evidence_source", "dataset"),
    )


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
