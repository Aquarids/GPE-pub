import json
import random
from pathlib import Path

from gpe.labels import parse_label
from gpe.poison import normalize_attack_type, normalize_poison_paths
from gpe.resources import GPE_DATA_PATH


class ClaimLoader:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else GPE_DATA_PATH
        self.records = read_jsonl(self.path)
        self.by_claim_id = {
            str(record["claim_id"]): record
            for record in self.records
            if record.get("claim_id")
        }

    def get_claim(self, claim_id, include_evidence=False):
        result = dict(self._record(claim_id))
        result.pop("evidence_environment", None)
        if include_evidence:
            result["evidence"] = self.benign_evidence(claim_id)
        else:
            result.pop("evidence", None)
        if "ground_truth" in result:
            result["ground_truth"] = parse_label(result["ground_truth"]).value
        return result

    def list_claims(self, category=None, label=None, include_evidence=False):
        expected_label = parse_label(label) if label is not None else None
        claims = []
        for record in self.records:
            if category is not None and record.get("category") != category:
                continue
            if expected_label is not None and parse_label(record.get("ground_truth")) != expected_label:
                continue
            claims.append(self.get_claim(record["claim_id"], include_evidence=include_evidence))
        return claims

    def decompose(self, claim_id):
        record = self._record(claim_id)
        return [dict(item) for item in record.get("subclaims") or record.get("subclaim_labels") or []]

    def benign_evidence(self, claim_id):
        record = self._record(claim_id)
        environment = record.get("evidence_environment") or {}
        return [
            dict(item)
            for item in environment.get("benign") or record.get("evidence") or []
        ]

    def related_distractors(self, claim_id):
        """Return entity-related, claim-irrelevant evidence for retrieval."""
        record = self._record(claim_id)
        environment = record.get("evidence_environment") or {}
        return [
            {**item, "evidence_type": "related_distractor", "contents": []}
            for item in environment.get("related_distractor") or []
        ]

    def _record(self, claim_id):
        key = str(claim_id)
        if key not in self.by_claim_id:
            raise KeyError(f"unknown claim_id: {claim_id}")
        return self.by_claim_id[key]


class DatasetEvidenceLoader:
    def __init__(self, data_path=None, poison_path=None):
        self.claim_loader = ClaimLoader(data_path)
        self.poison_path = poison_path
        self.poison_by_claim_id = self._load_poison(self.poison_path)

    def get_evidence_list(
        self,
        claim_id,
        top_k=None,
        poison_ratio=0.0,
        attack_type=None,
        seed=0,
        poison_cache=None,
        generate_missing_poison=False,
    ):
        benign = [
            copy_evidence(item)
            for item in self.claim_loader.benign_evidence(claim_id)
        ]
        ratio = clamp_ratio(poison_ratio)
        target_total = max(0, int(top_k) if top_k is not None else len(benign))
        if target_total == 0:
            return []
        if len(benign) < target_total:
            raise ValueError(
                f"insufficient benign evidence for {claim_id}: required={target_total} available={len(benign)}"
            )

        selected_benign = benign[:target_total]
        return self.mix_selected_evidence(
            claim_id,
            selected_benign,
            poison_ratio=ratio,
            attack_type=attack_type,
            seed=seed,
            poison_cache=poison_cache,
            generate_missing_poison=generate_missing_poison,
        )

    def mix_selected_evidence(
        self,
        claim_id,
        benign,
        poison_ratio=0.0,
        attack_type=None,
        seed=0,
        poison_cache=None,
        generate_missing_poison=False,
    ):
        """Replace selected benign records with poison while preserving list size."""
        record = self.claim_loader._record(claim_id)
        selected_benign = [copy_evidence(item) for item in benign]
        target_total = len(selected_benign)
        ratio = clamp_ratio(poison_ratio)
        if ratio <= 0.0 or target_total == 0:
            return selected_benign

        attack_type = normalize_attack_type(attack_type)
        poison_count = int(round(target_total * ratio))
        if poison_count <= 0:
            return selected_benign
        if poison_cache is not None:
            poison_cache.ensure(
                record,
                attack_type,
                poison_count,
                selected_benign,
                generate_missing=generate_missing_poison,
            )
            self.poison_by_claim_id = poison_cache.by_claim_id

        poison = self._poison_candidates(claim_id, attack_type)
        rng = random.Random(f"{seed}:{claim_id}:{attack_type or ''}:{ratio}:{target_total}")
        if attack_type == "ata":
            return self._replace_ata(selected_benign, poison, poison_count)
        if len(poison) < poison_count:
            raise ValueError(
                f"insufficient poison evidence for {claim_id} attack={attack_type}: "
                f"required={poison_count} available={len(poison)}"
            )
        positions = sorted(rng.sample(range(target_total), poison_count))
        poison_items = rng.sample(poison, poison_count)
        mixed = list(selected_benign)
        for position, poison_item in zip(positions, poison_items):
            mixed[position] = poison_item
        return mixed

    def _replace_ata(self, benign, poison, poison_count):
        by_source = {
            str(item.get("source_evidence_id")): item
            for item in poison
            if item.get("source_evidence_id")
        }
        mixed = list(benign)
        for position in range(poison_count):
            source_id = str(benign[position].get("evidence_id"))
            if source_id not in by_source:
                raise ValueError(
                    f"missing ATA replacement for source_evidence_id={source_id}"
                )
            mixed[position] = by_source[source_id]
        return mixed

    def _poison_candidates(self, claim_id, attack_type):
        attack_type = normalize_attack_type(attack_type)
        candidates = self.poison_by_claim_id.get(str(claim_id), [])
        return [
            copy_evidence(item)
            for item in candidates
            if attack_type is None
            or normalize_attack_type(item.get("attack_type")) == attack_type
        ]

    def _load_poison(self, path):
        by_claim_id = {}
        for record in self.claim_loader.records:
            claim_id = str(record.get("claim_id") or "")
            environment = record.get("evidence_environment") or {}
            poisoned = environment.get("poisoned") or {}
            for attack_type, items in poisoned.items():
                for item in items or []:
                    evidence = dict(item)
                    evidence.setdefault("claim_id", claim_id)
                    evidence.setdefault("attack_type", normalize_attack_type(attack_type))
                    by_claim_id.setdefault(claim_id, []).append(evidence)
        for poison_path in dict.fromkeys(normalize_poison_paths(path).values()):
            if not poison_path.exists():
                continue
            for record in read_jsonl(poison_path):
                claim_id = record.get("claim_id")
                if not claim_id:
                    continue
                items = record.get("evidence") if isinstance(record.get("evidence"), list) else [record]
                for item in items:
                    evidence = dict(item)
                    evidence.setdefault("claim_id", claim_id)
                    if record.get("attack_type") and not evidence.get("attack_type"):
                        evidence["attack_type"] = record["attack_type"]
                    by_claim_id.setdefault(str(claim_id), []).append(evidence)
        return by_claim_id


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
    return records


def copy_evidence(item):
    evidence = dict(item)
    evidence_type = str(evidence.get("evidence_type") or "").strip().lower()
    if evidence_type == "benign":
        evidence["poisoned"] = False
    elif evidence_type == "poisoned":
        evidence["poisoned"] = True
    else:
        evidence["poisoned"] = bool(evidence.get("attack_type"))
    return evidence


def clamp_ratio(value):
    try:
        ratio = float(value or 0.0)
    except (TypeError, ValueError):
        ratio = 0.0
    return max(0.0, min(1.0, ratio))
