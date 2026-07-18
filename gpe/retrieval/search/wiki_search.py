import time

import requests
import wikipediaapi
from wikipediaapi.exceptions import WikipediaException

from gpe.retrieval.document import Document
from gpe.retrieval.search.base_search import BaseSearch
from gpe.retrieval.search.content_filter import clean_content_blocks
from gpe.retrieval.search.content_filter import is_noise_section_title
from gpe.retrieval.search.errors import SearchProviderError


API_URL_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"


class WikiSearch(BaseSearch):
    NAMESPACE = "wiki"
    SOURCE_NAME = "wiki"

    def __init__(self, extra=None):
        super().__init__(extra=extra)
        self.language = self.extra.get("language", "en")
        self.user_agent = self.extra.get("user_agent", "GeoPoisonEval/0.1")
        self.rate_limit = self.extra.get("rate_limit", 0.72)
        self.timeout = self.extra.get("timeout", 15)
        auth_headers = self._build_auth_headers()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent, **auth_headers})
        if self.proxy:
            self.session.proxies.update(self.requests_proxies())
        wiki_kwargs = {
            "user_agent": self.user_agent,
            "language": self.language,
            "headers": auth_headers or None,
            "timeout": self.timeout,
        }
        if self.proxy:
            wiki_kwargs["proxy"] = self.proxy
            wiki_kwargs["trust_env"] = False
        self.wiki = wikipediaapi.Wikipedia(
            **wiki_kwargs,
        )

    def get_namespace(self):
        return self.NAMESPACE

    def get_source_name(self):
        return self.SOURCE_NAME

    def _cache_params(self, query, top_k, **kwargs):
        params = super()._cache_params(query, top_k, **kwargs)
        params["language"] = self.language
        return params

    def _build_auth_headers(self):
        access_token = self.extra.get("access_token")
        if not access_token:
            return {}
        return {"Authorization": f"Bearer {access_token}"}

    def _search(self, query, top_k) -> list[Document]:
        hits = self._api_search(query, top_k)
        documents = []
        for rank, hit in enumerate(hits, start=1):
            time.sleep(self.rate_limit)
            try:
                page = self._fetch_page(hit["title"])
            except SearchProviderError as error:
                self._record_error("page fetch failed", error)
                continue
            if page is None:
                continue
            documents.append(self._to_document(hit, page, rank))
        return documents

    def _api_search(self, query, top_k):
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": top_k,
            "format": "json",
        }
        try:
            response = self.session.get(
                API_URL_TEMPLATE.format(lang=self.language),
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            raise SearchProviderError("wiki", "api search request failed", error, query=query) from error
        if response.status_code != 200:
            raise SearchProviderError("wiki", f"api search returned HTTP {response.status_code}", query=query)
        try:
            payload = response.json()
        except ValueError as error:
            raise SearchProviderError("wiki", "api search returned invalid JSON", error, query=query) from error
        return payload.get("query", {}).get("search", [])

    def _fetch_page(self, title):
        try:
            page = self.wiki.page(title)
            if not page.exists():
                return None
            sections_text = _flatten_sections(page.sections)
            body = "\n".join(filter(None, [page.summary, sections_text]))
            paragraphs = clean_content_blocks(body.split("\n"))
            if not paragraphs:
                return None

            return {
                "title": page.title,
                "url": page.fullurl,
                "page_id": page.pageid,
                "summary": page.summary,
                "paragraphs": paragraphs,
                "section_titles": _section_titles(page.sections),
                "categories": list(page.categories.keys())[:20],
            }
        except (requests.RequestException, WikipediaException) as error:
            raise SearchProviderError("wiki", "page fetch request failed", error, title=title) from error

    def _to_document(self, hit, page, rank) -> Document:
        metadata = {
            "rank": rank,
            "page_id": page["page_id"],
            "language": self.language,
            "summary": page["summary"],
            "section_titles": page["section_titles"],
            "categories": page["categories"],
            "search_snippet": hit.get("snippet", ""),
            "wordcount": hit.get("wordcount"),
            "timestamp": hit.get("timestamp"),
        }
        return Document(
            document_id=_doc_id(page["page_id"]),
            contents=page["paragraphs"],
            url=page["url"],
            title=page["title"],
            summary=page["summary"],
            source_name=self.get_source_name(),
            locale=self.language,
            metadata=metadata,
        )


def _flatten_sections(sections, level=1):
    parts = []
    for section in sections:
        if is_noise_section_title(section.title):
            continue
        parts.append(f"\n{'#' * level} {section.title}\n")
        if section.text:
            parts.append(section.text)
        if section.sections:
            parts.append(_flatten_sections(section.sections, level=level + 1))
    return "\n".join(parts)


def _section_titles(sections):
    titles = []
    for section in sections:
        if is_noise_section_title(section.title):
            continue
        titles.append(section.title)
        if section.sections:
            titles.extend(_section_titles(section.sections))
    return titles


def _doc_id(page_id):
    return f"wiki-{page_id}"
