"""Neutral web search using the query supplied by the caller."""

from gpe.retrieval.search.ddg_base import DDGBaseSearch


class WebSearch(DDGBaseSearch):
    NAMESPACE = "web"
    SOURCE_NAME = "web"

    def get_namespace(self):
        return self.NAMESPACE

    def get_source_name(self):
        return self.SOURCE_NAME
