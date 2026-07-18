from gpe.retrieval.document import Document
from gpe.retrieval.search.ddg_base import DDGBaseSearch


class SiteRestrictedSearch(DDGBaseSearch):
    def get_namespace(self):
        return str(self.extra.get("namespace") or self.extra.get("source_name") or "site_restricted")

    def get_source_name(self):
        return str(self.extra.get("source_name") or self.get_namespace())

    def search(self, query, top_k=5, suffix=None, **kwargs) -> list[Document]:
        domains = self.extra.get("site_domains") or []
        if isinstance(domains, str):
            domains = [domains]
        site_terms = [f"site:{domain}" for domain in domains if str(domain).strip()]
        site_clause = [f"({' OR '.join(site_terms)})"] if len(site_terms) > 1 else site_terms
        extra_terms = list(self.extra.get("query_terms") or [])
        suffix = list(suffix or []) + site_clause + extra_terms
        documents = super().search(query, top_k=top_k, suffix=suffix, **kwargs)
        for document in documents:
            document.source_name = self.get_source_name()
            document.metadata.setdefault("site_restricted", True)
            document.metadata.setdefault("site_domains", list(domains))
            if self.extra.get("source_reputation") is not None:
                document.metadata.setdefault("source_reputation_hint", self.extra.get("source_reputation"))
            if self.extra.get("source_region"):
                document.metadata.setdefault("source_region", self.extra.get("source_region"))
        return documents
