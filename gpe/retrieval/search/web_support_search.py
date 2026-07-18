from gpe.retrieval.document import Document
from gpe.retrieval.search.ddg_base import DDGBaseSearch


class WebSupportSearch(DDGBaseSearch):
    NAMESPACE = "web_support"
    SOURCE_NAME = "web_support"
    DEFAULT_SUFFIX = ["confirmed OR proven OR true OR verified"]

    def get_namespace(self):
        return self.NAMESPACE

    def get_source_name(self):
        return self.SOURCE_NAME

    def search(self, query, top_k=5, suffix=None, **kwargs) -> list[Document]:
        if suffix is None:
            suffix = self.DEFAULT_SUFFIX
        return super().search(query, top_k=top_k, suffix=suffix, **kwargs)
