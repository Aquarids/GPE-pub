from gpe.retrieval.document import Document
from gpe.retrieval.search.ddg_base import DDGBaseSearch


class RedditSearch(DDGBaseSearch):
    NAMESPACE = "reddit"
    SOURCE_NAME = "reddit"
    DEFAULT_SUFFIX = ["site:reddit.com"]

    def get_namespace(self):
        return self.NAMESPACE

    def get_source_name(self):
        return self.SOURCE_NAME

    def search(self, query, top_k=5, suffix=None, **kwargs) -> list[Document]:
        if suffix is None:
            suffix = self.DEFAULT_SUFFIX
        kwargs.setdefault("allow_snippet_fallback", True)
        documents = super().search(query, top_k=top_k, suffix=suffix, **kwargs)
        for document in documents:
            document.source_name = self.get_source_name()
            document.metadata.setdefault("site_restricted", True)
            document.metadata.setdefault("site_domains", ["reddit.com"])
        return documents
