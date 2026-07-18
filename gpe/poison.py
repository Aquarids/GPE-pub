import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse


POISON_DOMAIN_PREFIXES = [
    "daily-trend",
    "insider-bulletin",
    "verified-scoop",
    "newswire-watch",
    "celebrity-ledger",
    "public-records-review",
]

POISON_TLDS = ["com", "net", "news", "co", "today"]
ATTACK_NAMES = ("fakegpt", "poisonedrag", "ata", "ignore")


def default_dynamic_poison_paths():
    return {
        attack: Path(f"output/dynamic_poison/malicious_{attack}.jsonl")
        for attack in ATTACK_NAMES
    }


def normalize_attack_type(value):
    value = str(value or "").strip().lower().replace("-", "_")
    if value == "poisoned_rag":
        return "poisonedrag"
    return value


def normalize_poison_paths(value):
    if not value:
        return {}
    if isinstance(value, dict):
        return {
            normalize_attack_type(key): Path(path)
            for key, path in value.items()
            if path
        }
    if isinstance(value, (list, tuple, set)):
        paths = {}
        for item in value:
            path = Path(item)
            normalized_stem = path.stem.lower().replace("_", "")
            attack = next(
                (name for name in ATTACK_NAMES if name in normalized_stem),
                "default",
            )
            paths[attack] = path
        return paths
    return {"default": Path(value)}


class PoisonCache:
    def __init__(self, path, dynamic_path=None, llm=None, logger=None):
        if logger is None and llm is not None:
            from gpe.helper.logger import ENV_LOCAL, Logger

            logger = Logger("poison-cache", ENV_LOCAL)
        self.prepared_paths = normalize_poison_paths(path)
        self.dynamic_paths = (
            normalize_poison_paths(dynamic_path)
            if dynamic_path is not None
            else default_dynamic_poison_paths()
        )
        self.llm = llm
        self.logger = logger
        self.records = []
        self.by_claim_id = {}
        self.indexed_ids = set()
        self.reload()

    def reload(self):
        self.records = []
        self.by_claim_id = {}
        self.indexed_ids = set()
        path_groups = (
            ("prepared", self.prepared_paths.values()),
            ("dynamic", self.dynamic_paths.values()),
        )
        for origin, paths in path_groups:
            for path in dict.fromkeys(paths):
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as file:
                    for line_number, line in enumerate(file, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"invalid poison JSONL at {path}:{line_number}: {error}"
                            ) from error
                        environment = record.get("evidence_environment") or {}
                        poisoned = environment.get("poisoned") or {}
                        if poisoned:
                            for attack_type, items in poisoned.items():
                                for item in items or []:
                                    evidence = dict(item)
                                    evidence.setdefault("claim_id", record.get("claim_id"))
                                    evidence.setdefault(
                                        "attack_type",
                                        normalize_attack_type(attack_type),
                                    )
                                    self._index(evidence, origin=origin)
                        elif record.get("attack_type") or record.get("poisoned"):
                            self._index(record, origin=origin)

    def candidates(self, claim_id, attack_type):
        attack_type = normalize_attack_type(attack_type)
        return [
            dict(item)
            for item in self.by_claim_id.get(str(claim_id), [])
            if normalize_attack_type(item.get("attack_type")) == attack_type
        ]

    def ensure(self, claim, attack_type, required, benign, generate_missing=True):
        attack_type = normalize_attack_type(attack_type)
        if attack_type not in ATTACK_NAMES:
            raise ValueError(f"unknown attack_type: {attack_type}")
        if required <= 0:
            return []
        if attack_type == "ata":
            return self._ensure_ata(claim, required, benign, generate_missing)
        existing = self.candidates(claim["claim_id"], attack_type)
        if len(existing) < required:
            if not generate_missing:
                self._raise_insufficient(claim, attack_type, required, len(existing))
            self._generate_batch(claim, attack_type, required - len(existing), benign, existing)
        return self.candidates(claim["claim_id"], attack_type)

    def _ensure_ata(self, claim, required, benign, generate_missing):
        sources = list(benign[:required])
        existing = self.candidates(claim["claim_id"], "ata")
        by_source = {
            str(item.get("source_evidence_id")): item
            for item in existing
            if item.get("source_evidence_id")
        }
        missing = [
            source for source in sources
            if str(source.get("evidence_id")) not in by_source
        ]
        if missing and not generate_missing:
            self._raise_insufficient(claim, "ata", required, required - len(missing))
        if missing:
            self._generate_ata(claim, missing)
        return self.candidates(claim["claim_id"], "ata")

    def _generate_batch(self, claim, attack_type, count, benign, existing):
        attack = build_attack(attack_type, self.llm, self.logger)
        generated = attack.generate_poison_contents(
            query=claim["original_claim"],
            label=claim["ground_truth"],
            n_content=count,
            category=claim.get("category"),
        )
        if len(generated) != count:
            raise RuntimeError(
                f"{attack_type} generated {len(generated)} items for {claim['claim_id']}; expected {count}"
            )
        start = len(existing) + 1
        references = list(benign) or [{}]
        for offset, item in enumerate(generated):
            index = start + offset
            evidence_id = generated_id(attack_type, claim["claim_id"], index)
            reference = references[(index - 1) % len(references)]
            self._append(build_record(claim, attack_type, item, evidence_id, domain_reference=reference))

    def _generate_ata(self, claim, sources):
        attack = build_attack("ata", self.llm, self.logger)
        for index, source in enumerate(sources, start=1):
            source_id = source.get("evidence_id") or f"index-{index:04d}"
            evidence_id = generated_id("ata", claim["claim_id"], index, source)
            reference_content = normalize_content(source.get("contents"))
            generated = attack.generate_poison_contents(
                query=claim["original_claim"],
                label=claim["ground_truth"],
                n_content=1,
                category=claim.get("category"),
                extra={"reference_content": reference_content},
            )
            if len(generated) != 1:
                raise RuntimeError(
                    f"ata generated {len(generated)} items for {claim['claim_id']}:{source_id}; expected 1"
                )
            self._append(
                build_record(
                    claim,
                    "ata",
                    generated[0],
                    evidence_id,
                    source_evidence_id=source_id,
                    domain_reference=source,
                )
            )

    def _append(self, record):
        attack_type = normalize_attack_type(record.get("attack_type"))
        path = self.dynamic_paths.get(attack_type) or self.dynamic_paths.get("default")
        if path is None:
            raise ValueError(
                f"dynamic poison output path is required for attack_type={attack_type}"
            )
        record = dict(record)
        record["poison_origin"] = "dynamic"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            file.flush()
        self._index(record, origin="dynamic")

    def _index(self, record, origin=None):
        claim_id = record.get("claim_id")
        if not claim_id:
            return
        items = record.get("evidence") if isinstance(record.get("evidence"), list) else [record]
        for item in items:
            evidence = dict(item)
            evidence.setdefault("claim_id", claim_id)
            evidence_id = evidence.get("evidence_id")
            if evidence_id and evidence_id in self.indexed_ids:
                continue
            evidence["poison_origin"] = evidence.get("poison_origin") or origin
            evidence["attack_type"] = normalize_attack_type(
                evidence.get("attack_type") or record.get("attack_type")
            )
            self.records.append(evidence)
            self.by_claim_id.setdefault(str(claim_id), []).append(evidence)
            if evidence_id:
                self.indexed_ids.add(evidence_id)

    def _raise_insufficient(self, claim, attack_type, required, available):
        raise ValueError(
            f"insufficient poison evidence for {claim['claim_id']} attack={attack_type}: "
            f"required={required} available={available}; enable dynamic generation or prefill the cache"
        )


def build_attack(name, llm, logger):
    if name == "ignore":
        from gpe.attack.ignore_injection import IgnoreInjection

        return IgnoreInjection(logger, config={})
    if llm is None:
        raise ValueError("dynamic poison generation requires an LLMWrapper instance")
    if name == "fakegpt":
        from gpe.attack.fakegpt import FakeGPT

        return FakeGPT(llm, logger)
    if name == "poisonedrag":
        from gpe.attack.poisoned_rag import PoisonedRAG

        return PoisonedRAG(llm, logger, config={})
    if name == "ata":
        from gpe.attack.ata import AdaptiveTamperingAttack

        return AdaptiveTamperingAttack(llm, logger, config={})
    raise KeyError(name)


def generated_id(attack_name, claim_id, index, benign=None):
    if attack_name == "ata":
        source_id = benign.get("evidence_id") or f"index-{index:04d}"
        return f"malicious-ata-{slugify(claim_id)}-{slugify(source_id)}"
    return f"malicious-{attack_name}-{slugify(claim_id)}-{index:04d}"


def build_record(
    claim,
    attack_name,
    generated,
    evidence_id,
    source_evidence_id=None,
    domain_reference=None,
):
    content = normalize_content(generated.get("content") or generated.get("contents"))
    if not content:
        raise ValueError(f"{evidence_id}: generated object has no content")
    title = normalize_content(generated.get("title")) or content[:100]
    source = normalize_content(generated.get("source")) or "Independent Bulletin"
    reference_host = source_host(domain_reference)
    domain = poison_domain(reference_host or source, evidence_id)
    record = {
        "claim_id": str(claim["claim_id"]),
        "evidence_id": evidence_id,
        "evidence_type": "poisoned",
        "attack_type": normalize_attack_type(attack_name),
        "contents": [content],
        "title": title,
        "summary": content[:300],
        "url": f"https://{domain}/{slugify(generated.get('name') or title)}",
        "source_name": poisoned_source_name(claim, generated),
        "locale": "global",
        "author": normalize_content(generated.get("author")),
        "published_at": normalize_content(
            generated.get("publish_date") or generated.get("published_at")
        ),
        "target_answer": normalize_content(generated.get("target_answer")),
        "simulated_domain": domain,
        "controlled_benchmark": True,
    }
    if generated.get("target_label") is not None:
        record["target_label"] = normalize_content(generated.get("target_label"))
    if reference_host:
        record["simulated_reference_host"] = reference_host
    if source_evidence_id:
        record["source_evidence_id"] = str(source_evidence_id)
    return record


def poison_domain(source, evidence_id):
    digest = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()
    prefix = POISON_DOMAIN_PREFIXES[int(digest[:2], 16) % len(POISON_DOMAIN_PREFIXES)]
    tld = POISON_TLDS[int(digest[2:4], 16) % len(POISON_TLDS)]
    source_slug = slugify(source).replace("-", "")[:18] or "independentdesk"
    return f"{prefix}-{source_slug}.{tld}"


def poisoned_source_name(claim, generated):
    labels = {"true", "mostly_true", "half_true", "mostly_false", "false", "uncertain"}
    explicit = str(generated.get("target_label") or "").strip().lower().replace("-", "_")
    if explicit not in labels:
        target_answer = str(generated.get("target_answer") or "").strip().lower().replace("-", "_")
        explicit = target_answer if target_answer in labels else ""
    if not explicit:
        ground_truth = str(claim.get("ground_truth") or "").strip().lower().replace("-", "_")
        explicit = "true" if ground_truth in {"false", "mostly_false"} else "false"
    return "web_support" if explicit in {"true", "mostly_true"} else "web_refute"


def slugify(value):
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return text[:80] or "record"


def normalize_content(value):
    if isinstance(value, list):
        value = " ".join(str(item).strip() for item in value if str(item).strip())
    return re.sub(r"\s+", " ", str(value or "")).strip()


def source_host(evidence):
    if not evidence:
        return ""
    host = urlparse(str(evidence.get("url") or "")).hostname or ""
    return host.lower().removeprefix("www.")
