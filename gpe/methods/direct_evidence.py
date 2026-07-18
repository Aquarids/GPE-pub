import json
from typing import Any, Dict

from gpe.methods.base import BaseDetector
from gpe.methods.label_utils import normalize_compare_label
from gpe.retrieval.search.web_support_search import WebSupportSearch
from gpe.utils.json_utils import extract_json


class DirectEvidenceDetector(BaseDetector):
    """Judge a claim from the claim and the contents of all supplied evidence items."""

    SYSTEM_PROMPT = """You are an evidence-grounded fact-verification system.
Judge the input claim only from the supplied evidence contents. Return JSON only:
{
  "verdict": "true|mostly_true|half_true|mostly_false|false|uncertain",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}
Use uncertain when the supplied evidence is insufficient or highly conflicting.
"""

    def __init__(self, logger, llm, config=None):
        super().__init__(logger, llm, config)
        self.max_urls = int(self.config.get("max_urls", 5))
        self.web_support = WebSupportSearch(extra={
            "use_cache": bool(self.config.get("use_cache", True)),
            "cache_dir": self.config.get("cache_dir", "cache/compare"),
        })

    def _search(self, claim):
        documents = []
        for item in self.web_support.search(claim, top_k=self.max_urls):
            documents.append({
                "title": item.title or "",
                "source_name": item.metadata.get("source_name", ""),
                "url": item.url or "",
                "published_at": item.metadata.get("published_at"),
                "contents": [item.text],
                "content": item.text,
            })
        return documents

    def detect(self, claim: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        evidence = self.resolve_evidence(claim, meta, self._search)
        evidence_items = [
            {
                "title": item.get("title"),
                "source_name": item.get("source_name"),
                "published_at": item.get("published_at"),
                "author": item.get("author"),
                "url": item.get("url"),
                "contents": item.get("contents") or [],
            }
            for item in evidence
        ]
        response = self.llm.dialogue(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Claim:\n{claim}\n\nEvidence items:\n"
                        f"{json.dumps(evidence_items, ensure_ascii=False)}"
                    ),
                },
            ],
            stream=True,
        )
        payload = extract_json(response)
        verdict = str(payload.get("verdict") or "uncertain").strip().lower().replace("-", "_")
        if verdict != "uncertain":
            verdict = normalize_compare_label(verdict)
        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": max(0.0, min(1.0, float(payload.get("confidence", 0.0)))),
            "explanation": str(payload.get("reasoning") or "").strip(),
            "method": "DirectEvidence",
            "extra": {"evidence_count": len(evidence_items)},
        }
