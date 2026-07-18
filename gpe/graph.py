import json
import re
from collections import defaultdict
from pathlib import Path

from gpe.resources import GRAPH_DATA_DIR


def _normalize(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _read_jsonl(path, key):
    rows = {}
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                rows[str(row[key])] = row
    return rows


class GPEKnowledgeGraph:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else GRAPH_DATA_DIR
        self.entities = _read_jsonl(self.data_dir / "entities.jsonl", "entity_id")
        self.events = _read_jsonl(self.data_dir / "events.jsonl", "event_id")
        self.sources = _read_jsonl(self.data_dir / "sources.jsonl", "source_id")
        self.claims = _read_jsonl(self.data_dir / "claims.jsonl", "claim_id")
        self.entity_aliases = defaultdict(list)
        for entity_id, entity in self.entities.items():
            for alias in entity.get("aliases") or [entity.get("name", "")]:
                key = _normalize(alias)
                if key:
                    self.entity_aliases[key].append(entity_id)

    def statistics(self):
        return {
            "claims": len(self.claims),
            "entities": len(self.entities),
            "events": len(self.events),
            "sources": len(self.sources),
        }

    def get_entity(self, entity_id):
        return self.entities.get(entity_id)

    def find_entities(self, query, limit=10):
        if limit < 1:
            return []
        key = _normalize(query)
        entity_ids = list(dict.fromkeys(self.entity_aliases.get(key, [])))
        if len(entity_ids) < limit and key:
            for entity_id, entity in self.entities.items():
                if entity_id in entity_ids:
                    continue
                aliases = entity.get("aliases") or [entity.get("name", "")]
                if any(key in _normalize(alias) for alias in aliases):
                    entity_ids.append(entity_id)
                    if len(entity_ids) == limit:
                        break
        return [self.entities[entity_id] for entity_id in entity_ids[:limit]]

    def get_event(self, event_id):
        return self.events.get(event_id)

    def find_events(self, query, limit=10):
        key = _normalize(query)
        return [
            event for event in self.events.values()
            if key and any(key in _normalize(text)
                           for text in [event.get("description", ""), *(event.get("aliases") or [])])
        ][:limit]

    def get_source(self, source_id):
        return self.sources.get(source_id)

    def find_sources(self, query, limit=10):
        key = _normalize(query)
        return [source for source in self.sources.values()
                if key and key in _normalize(source.get("name", ""))][:limit]

    def get_claim(self, claim_id):
        return self.claims.get(claim_id)

    def claim_context(self, claim_id):
        claim = self.get_claim(claim_id)
        if not claim:
            return None
        entity_ids = set((claim.get("mentions") or {}).values())
        event_ids = {item.get("event_id") for item in claim.get("subclaims") or []}
        source_ids = {item.get("source_id") for item in claim.get("evidence") or []}
        for evidence in claim.get("evidence") or []:
            entity_ids.update(evidence.get("entity_ids") or [])
        return {
            "claim": claim,
            "entities": [self.entities[item] for item in sorted(entity_ids) if item in self.entities],
            "events": [self.events[item] for item in sorted(event_ids) if item in self.events],
            "sources": [self.sources[item] for item in sorted(source_ids) if item in self.sources],
        }
