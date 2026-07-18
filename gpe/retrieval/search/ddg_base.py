import hashlib

from gpe.helper.article import Article
from gpe.helper.ddg import DDG
from gpe.retrieval.document import Document
from gpe.retrieval.search.base_search import BaseSearch
from gpe.retrieval.search.errors import SearchProviderError


class DDGBaseSearch(BaseSearch):
    def __init__(self, extra=None):
        super().__init__(extra=extra)
        self.ddg = DDG(
            region=self.extra.get("ddg_region", "wt-wt"),
            safesearch=self.extra.get("ddg_safesearch", "moderate"),
            proxy=self.proxy,
            backend=self.extra.get("ddg_backend", "duckduckgo,brave,yahoo"),
        )
        self.article = Article(
            timeout=self.extra.get("article_timeout", 10),
            language=self.extra.get("article_language"),
            proxy=self.proxy,
        )

    def _search(self, query, top_k, suffix=None, **kwargs) -> list[Document]:
        suffix = list(suffix or [])
        full_query = self._build_query(query, suffix)
        hits = self.ddg.search(full_query, top_k=top_k)
        if self.ddg.last_error:
            self._append_last_error(self.ddg.last_error)

        documents = []
        fetch_errors = []
        for rank, hit in enumerate(hits, start=1):
            url = hit["url"]
            try:
                fetched = self.article.fetch(url)
            except SearchProviderError as error:
                if self.extra.get("allow_snippet_fallback"):
                    documents.append(self._to_snippet_document(hit, rank, suffix, str(error)))
                    continue
                if len(fetch_errors) < 3:
                    fetch_errors.append(str(error))
                continue
            if fetched is None:
                if self.extra.get("allow_snippet_fallback"):
                    documents.append(self._to_snippet_document(hit, rank, suffix, "empty article text"))
                    continue
                if len(fetch_errors) < 3:
                    fetch_errors.append(f"{url} -> empty article text")
                continue
            documents.append(self._to_document(hit, fetched, rank, suffix))
        if hits and not documents and fetch_errors:
            self._append_last_error(
                "article fetch failed for all hits: " + "; ".join(fetch_errors)
            )
        return documents

    def _append_last_error(self, message):
        if not message:
            return
        if not self.last_error:
            self.last_error = message
            return
        self.last_error = f"{self.last_error}; {message}"

    def _build_query(self, query, suffix):
        return " ".join([query, *suffix])

    def _to_document(self, hit, fetched, rank, suffix) -> Document:
        url = hit["url"]
        metadata = {
            "rank": rank,
            "snippet": hit.get("snippet", ""),
            "authors": fetched.get("authors", []),
            "publish_date": fetched.get("publish_date"),
            "top_image": fetched.get("top_image", ""),
            "query_suffix": list(suffix),
        }
        return Document(
            document_id=_doc_id(url),
            contents=fetched.get("paragraphs", []),
            url=url,
            title=fetched.get("title") or hit.get("title", ""),
            summary=hit.get("snippet", ""),
            source_name=self.get_source_name(),
            locale="global",
            metadata=metadata,
        )

    def _to_snippet_document(self, hit, rank, suffix, fetch_error) -> Document:
        url = hit["url"]
        snippet = hit.get("snippet", "")
        metadata = {
            "rank": rank,
            "snippet": snippet,
            "query_suffix": list(suffix),
            "snippet_fallback": True,
            "fetch_error": fetch_error,
        }
        contents = [snippet] if snippet else []
        return Document(
            document_id=_doc_id(url),
            contents=contents,
            url=url,
            title=hit.get("title", ""),
            summary=snippet,
            source_name=self.get_source_name(),
            locale="global",
            metadata=metadata,
        )


def _doc_id(url):
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"web-{digest}"
