import hashlib
import logging
import json
import re
from pathlib import Path

from gpe.retrieval.document import Document


LOGGER = logging.getLogger(__name__)


class SearchPoisoner:
    def __init__(self, path, ratio=0.0, seed=2026, min_overlap=0.35, mode="prepend"):
        self.path = Path(path) if path else None
        self.ratio = max(0.0, min(1.0, float(ratio or 0.0)))
        self.seed = int(seed)
        self.min_overlap = max(0.0, min(1.0, float(min_overlap or 0.0)))
        self.mode = str(mode or "prepend").strip().lower()
        self.records = []
        self.by_statement = {}
        self.by_id = {}
        self.forced_record = None
        if self.path and self.path.exists() and self.ratio > 0.0:
            self._load()

    @property
    def enabled(self):
        return bool(self.records) and self.ratio > 0.0

    def maybe_mix(self, query, documents, top_k, source_name):
        documents = list(documents or [])
        if not self.enabled:
            return documents
        record = self.forced_record or self.match(query)
        if record is None:
            return documents
        target_total = max(1, int(top_k or len(documents) or 1))
        poison_count = self._poison_count(target_total)
        if poison_count <= 0:
            return documents[:target_total]
        poison_docs = [self.to_document(record, source_name, variant=index) for index in range(poison_count)]
        clean_limit = max(0, target_total - poison_count)
        clean_docs = documents[:clean_limit]
        LOGGER.info(
            "search poison injected source=%s ratio=%s poison_docs=%s clean_docs=%s returned_docs=%s query=%s",
            source_name,
            self.ratio,
            len(poison_docs),
            len(clean_docs),
            len(poison_docs) + len(clean_docs),
            str(query or "")[:120],
        )
        if self.mode == "replace":
            return poison_docs + clean_docs
        return poison_docs + clean_docs

    def match(self, query):
        text = normalize_text(query)
        if not text:
            return None
        if text in self.by_statement:
            return self.by_statement[text]
        if text in self.by_id:
            return self.by_id[text]
        best = None
        best_score = 0.0
        query_tokens = token_set(text)
        if not query_tokens:
            return None
        for record in self.records:
            for key in record["match_texts"]:
                if key and (key in text or text in key):
                    return record
                score = overlap_score(query_tokens, record["tokens"])
                if score > best_score:
                    best = record
                    best_score = score
        return best if best_score >= self.min_overlap else None

    def force_record_for_text(self, value):
        self.forced_record = self.match(value)
        return self.forced_record is not None

    def clear_forced_record(self):
        self.forced_record = None

    def to_document(self, record, source_name, variant=0):
        raw = record["raw"]
        source = clean(raw.get("poison_source")) or "Poisoned Search Result"
        title = clean(raw.get("poison_title")) or clean(raw.get("statement")) or "Poisoned Search Result"
        content = clean(raw.get("poison_content")) or clean(raw.get("poisoned_claim"))
        publish_date = clean(raw.get("poison_publish_date"))
        author = clean(raw.get("poison_author"))
        source_sample_id = clean(raw.get("source_sample_id"))
        attack = clean(raw.get("attack"))
        doc_key = clean(raw.get("poison_name")) or source_sample_id or title
        variant_key = f"{doc_key}|mirror:{variant}"
        digest = hashlib.sha1(variant_key.encode("utf-8")).hexdigest()[:12]
        domain = poison_domain(source, digest)
        slug = slugify(clean(raw.get("poison_name")) or title)
        url = f"https://{domain}/{slug}-{digest[:8]}"
        metadata = {
            "rank": 0,
            "poisoned": True,
            "poison_attack": attack,
            "poison_source": source,
            "poison_source_sample_id": source_sample_id,
            "poison_mirror_index": int(variant),
            "source_type_hint": "minor_media",
            "official_source_hint": False,
            "source_credibility_hint": 0.25,
            "authors": [author] if author else [],
            "publish_date": publish_date or None,
            "snippet": content[:280],
        }
        return Document(
            document_id=f"poison-{digest}",
            contents=[content] if content else [title],
            url=url,
            title=title,
            summary=content[:300],
            source_name=source_name,
            locale="global",
            credibility=0.25,
            metadata=metadata,
            extra={"poisoned": True, "poison_source": source, "simulated_domain": domain},
        )

    def _load(self):
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                match_texts = [
                    normalize_text(raw.get("statement")),
                    normalize_text(raw.get("attack_query")),
                    normalize_text(raw.get("poisoned_claim")),
                    normalize_text(raw.get("poison_title")),
                    normalize_text(raw.get("poison_target_answer")),
                ]
                match_texts = [text for text in match_texts if text]
                tokens = set()
                for text in match_texts:
                    tokens.update(token_set(text))
                record = {"raw": raw, "match_texts": match_texts, "tokens": tokens}
                self.records.append(record)
                for text in match_texts[:2]:
                    self.by_statement[text] = record
                for key in record_ids(raw):
                    self.by_id[normalize_text(key)] = record

    def _poison_count(self, target_total):
        if self.ratio <= 0.0:
            return 0
        return min(target_total, int(round(self.ratio * target_total)))


def build_poisoner(extra):
    if not extra or not extra.get("search_poison_enabled"):
        return None
    return SearchPoisoner(
        path=extra.get("search_poison_path"),
        ratio=extra.get("search_poison_ratio", 0.0),
        seed=extra.get("search_poison_seed", 2026),
        min_overlap=extra.get("search_poison_min_overlap", 0.35),
        mode=extra.get("search_poison_mode", "prepend"),
    )


def record_ids(raw):
    values = [
        raw.get("id"),
        raw.get("sample_id"),
        raw.get("source_sample_id"),
        raw.get("poison_name"),
    ]
    parts = [raw.get("attack"), raw.get("source_dataset"), raw.get("source_sample_id"), raw.get("poison_name")]
    values.append("-".join(str(part) for part in parts if part))
    return [str(value) for value in values if str(value or "").strip()]


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def clean(value):
    return str(value or "").strip()


def token_set(value):
    return {token for token in re.findall(r"[a-z0-9]+", str(value).lower()) if len(token) > 2}


def overlap_score(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


POISON_DOMAIN_PREFIXES = [
    "daily-trend",
    "insider-bulletin",
    "verified-scoop",
    "newswire-watch",
    "celebrity-ledger",
    "public-records-review",
]

POISON_TLDS = ["com", "net", "news", "co", "today"]


def poison_domain(source, digest):
    source_slug = slugify(source).replace("-", "")[:18] or "exclusivewire"
    prefix = POISON_DOMAIN_PREFIXES[int(digest[:2], 16) % len(POISON_DOMAIN_PREFIXES)]
    tld = POISON_TLDS[int(digest[2:4], 16) % len(POISON_TLDS)]
    # Simulate attacker-controlled small or typosquat-looking domains.
    return f"{prefix}-{source_slug}.{tld}"


def slugify(value):
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return text[:80] or "story"
