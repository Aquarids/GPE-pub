import arxiv

from gpe.retrieval.document import Document
from gpe.retrieval.search.base_search import BaseSearch
from gpe.retrieval.search.errors import SearchProviderError


class ArxivSearch(BaseSearch):
    NAMESPACE = "arxiv"
    SOURCE_NAME = "arxiv"

    def __init__(self, extra=None):
        super().__init__(extra=extra)
        self.sort_by = self.extra.get("sort_by", arxiv.SortCriterion.Relevance)
        self.sort_order = self.extra.get("sort_order", arxiv.SortOrder.Descending)
        self.client = arxiv.Client(
            page_size=self.extra.get("page_size", 100),
            delay_seconds=self.extra.get("rate_limit", 3.0),
            num_retries=self.extra.get("num_retries", 3),
        )

    def get_namespace(self):
        return self.NAMESPACE

    def get_source_name(self):
        return self.SOURCE_NAME

    def _cache_params(self, query, top_k, **kwargs):
        params = super()._cache_params(query, top_k, **kwargs)
        params["sort_by"] = str(kwargs.get("sort_by", self.sort_by))
        params["sort_order"] = str(kwargs.get("sort_order", self.sort_order))
        return params

    def _search(self, query, top_k, sort_by=None, sort_order=None) -> list[Document]:
        search = arxiv.Search(
            query=query,
            max_results=top_k,
            sort_by=sort_by or self.sort_by,
            sort_order=sort_order or self.sort_order,
        )
        documents = []
        try:
            iterator = iter(self.client.results(search))
        except Exception as error:
            raise SearchProviderError("arxiv", "search request failed", error, query=query) from error

        rank = 1
        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except Exception as error:
                raise SearchProviderError("arxiv", "search request failed", error, query=query) from error
            documents.append(self._to_document(result, rank))
            rank += 1
        return documents

    def _to_document(self, result, rank) -> Document:
        arxiv_id = result.get_short_id()
        metadata = {
            "rank": rank,
            "arxiv_id": arxiv_id,
            "authors": [author.name for author in result.authors],
            "primary_category": result.primary_category,
            "categories": result.categories,
            "published": result.published.isoformat() if result.published else None,
            "updated": result.updated.isoformat() if result.updated else None,
            "pdf_url": result.pdf_url,
            "doi": result.doi,
            "journal_ref": result.journal_ref,
            "comment": result.comment,
        }
        return Document(
            document_id=_doc_id(arxiv_id),
            contents=[result.summary.strip()],
            url=result.entry_id,
            title=result.title,
            summary=result.summary,
            source_name=self.get_source_name(),
            locale="global",
            metadata=metadata,
        )


def _doc_id(arxiv_id):
    return f"arxiv-{arxiv_id}"
