import logging

from gpe.helper.search_cache import SearchCache
from gpe.retrieval.document import Document
from gpe.retrieval.search.errors import SearchProviderError
from gpe.retrieval.search.poisoned_search import build_poisoner


LOGGER = logging.getLogger(__name__)


class BaseSearch:
    def __init__(self, extra=None):
        self.extra = dict(extra or {})
        self.last_error = ""
        self.proxy = self.extra.get("proxy") or None
        self.poisoner = build_poisoner(self.extra)
        self.cache = self._init_cache()

    def get_namespace(self) -> str:
        raise NotImplementedError

    def get_source_name(self) -> str:
        raise NotImplementedError

    def search(self, query, top_k=5, **kwargs) -> list[Document]:
        self.last_error = ""
        params = self._cache_params(query, top_k, **kwargs)
        if self.cache is not None:
            cached = self.cache.get(self.get_namespace(), params)
            if cached is not None:
                documents = [Document.from_dict(item) for item in cached]
                return self._apply_poison(query, documents, top_k)

        try:
            documents = self._search(query, top_k, **kwargs)
        except Exception as error:
            self._record_error("search failed", error)
            return self._apply_poison(query, [], top_k)

        if self.cache is not None and not (self.last_error and not documents):
            self.cache.put(
                self.get_namespace(),
                params,
                [doc.to_dict() for doc in documents],
            )
        return self._apply_poison(query, documents, top_k)

    def _apply_poison(self, query, documents, top_k):
        if self.poisoner is None:
            return documents
        for key in ("search_poison_match_id", "search_poison_match_text", "root_statement", "root_claim_id"):
            value = self.extra.get(key)
            if value and hasattr(self.poisoner, "force_record_for_text"):
                if self.poisoner.force_record_for_text(value):
                    break
        return self.poisoner.maybe_mix(query, documents, top_k, self.get_source_name())

    def _record_error(self, context, error):
        message = f"{context}: {type(error).__name__}: {error}"
        self.last_error = _append_error(self.last_error, message)
        LOGGER.warning("%s source=%s", message, self.get_source_name())

    def _init_cache(self):
        if not self.extra.get("use_cache"):
            return None
        namespace = self.get_namespace()
        if not namespace:
            raise ValueError(
                f"{type(self).__name__}.get_namespace() must return a non-empty string when use_cache is True"
            )
        return SearchCache(self.extra["cache_dir"])

    def _cache_params(self, query, top_k, **kwargs):
        params = {"query": query, "top_k": top_k}
        params.update(kwargs)
        return params

    def _search(self, query, top_k, **kwargs) -> list[Document]:
        raise NotImplementedError

    def requests_proxies(self):
        if not self.proxy:
            return None
        return {"http": self.proxy, "https": self.proxy}


def _append_error(existing, message):
    if not existing:
        return message
    return f"{existing}; {message}"
