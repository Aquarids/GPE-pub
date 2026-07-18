from abc import ABC, abstractmethod
from typing import Dict, Any, List

from gpe.helper.llm_wrapper import LLMWrapper
from gpe.helper.logger import Logger

class BaseDetector(ABC):

    def __init__(self, logger: Logger, llm: LLMWrapper, config=None):
        super().__init__()
        self.logger = logger
        self.llm = llm
        self.config = config or {}
        self.evidence_source = str(self.config.get("evidence_source", "dataset")).strip().lower()
        if self.evidence_source not in {"dataset", "search"}:
            raise ValueError("evidence_source must be 'dataset' or 'search'")

    def provided_evidence(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        documents = []
        for item in (meta or {}).get("evidence", []) or []:
            contents = item.get("contents") or []
            if isinstance(contents, str):
                contents = [contents]
            contents = [str(value) for value in contents if value]
            documents.append({
                "title": item.get("title") or "",
                "source_name": item.get("source_name") or "",
                "url": item.get("url") or "",
                "published_at": item.get("published_at"),
                "author": item.get("author"),
                "contents": contents,
                "content": "\n".join(contents),
            })
        return documents

    def resolve_evidence(self, query: str, meta: Dict[str, Any], search_fn):
        if self.evidence_source == "dataset":
            return self.provided_evidence(meta)
        return search_fn(query)
        
    def attach_common_fields(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return result
        
    @abstractmethod
    def detect(self, claim: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def detect_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            content = item.get('content', '')
            meta = item.get('meta', {})
            result = self.detect(content, meta)
            results.append(self.attach_common_fields(result))
        return results
