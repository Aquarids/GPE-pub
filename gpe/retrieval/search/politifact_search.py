import hashlib
import time

import requests
from bs4 import BeautifulSoup

from gpe.retrieval.document import Document
from gpe.retrieval.search.base_search import BaseSearch
from gpe.retrieval.search.content_filter import clean_content_blocks
from gpe.retrieval.search.errors import SearchProviderError


BASE_URL = "https://www.politifact.com"
SEARCH_URL = "https://www.politifact.com/search/"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class PolitifactSearch(BaseSearch):
    NAMESPACE = "politifact"
    SOURCE_NAME = "politifact"

    def __init__(self, extra=None):
        super().__init__(extra=extra)
        self.timeout = self.extra.get("timeout", 15)
        self.rate_limit = self.extra.get("rate_limit", 1.5)
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        if self.proxy:
            self.session.proxies.update(self.requests_proxies())

    def get_namespace(self):
        return self.NAMESPACE

    def get_source_name(self):
        return self.SOURCE_NAME

    def _search(self, query, top_k) -> list[Document]:
        try:
            response = self.session.get(SEARCH_URL, params={"q": query}, timeout=self.timeout)
        except requests.RequestException as error:
            raise SearchProviderError("politifact", "search request failed", error, query=query) from error
        if response.status_code != 200:
            raise SearchProviderError("politifact", f"search returned HTTP {response.status_code}", query=query)

        items = self._parse_search_page(response.text)
        documents = []
        for rank, item in enumerate(items[:top_k], start=1):
            time.sleep(self.rate_limit)
            try:
                article = self._fetch_article(item["url"])
            except SearchProviderError as error:
                self._record_error("article fetch failed", error)
                continue
            if article is None:
                continue
            documents.append(self._to_document(item, article, rank))
        return documents

    def _parse_search_page(self, html):
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select("div.o-listease__item"):
            container = item.select_one("div.m-result")
            if container is None:
                continue
            link = container.select_one("div.c-textgroup__title a")
            if link is None:
                continue
            href = link.get("href", "")
            url = href if href.startswith("http") else f"{BASE_URL}{href}"
            results.append(
                {
                    "url": url,
                    "title": link.get_text(strip=True),
                    "author_info": _text(container, "div.c-textgroup__author"),
                    "meta_info": _text(container, "div.c-textgroup__meta"),
                    "verdict": _list_verdict(container),
                }
            )
        return results

    def _fetch_article(self, url):
        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as error:
            raise SearchProviderError("politifact", "article request failed", error, url=url) from error
        if response.status_code != 200:
            raise SearchProviderError("politifact", f"article returned HTTP {response.status_code}", url=url)

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "svg", "iframe"]):
            tag.decompose()

        paragraphs = _extract_paragraphs(soup)
        if not paragraphs:
            return None

        return {
            "title": _text(soup, "h1.c-title") or (soup.title.get_text(strip=True) if soup.title else ""),
            "paragraphs": paragraphs,
            "verdict": _article_verdict(soup),
            "statement": _text(soup, "div.m-statement__quote"),
            "speaker": _text(soup, "div.m-statement__name"),
            "speaker_desc": _text(soup, "div.m-statement__desc"),
            "research_date": _text(soup, "div.m-author__date"),
            "author": _text(soup, "div.m-author__content a.m-author__name"),
        }

    def _to_document(self, item, article, rank) -> Document:
        url = item["url"]
        contents = self._document_contents(article)
        metadata = {
            "rank": rank,
            "search_snippet": f"{item['author_info']} {item['meta_info']}".strip(),
            "search_verdict": item["verdict"],
            "verdict": article["verdict"],
            "statement": article["statement"],
            "speaker": article["speaker"],
            "speaker_desc": article["speaker_desc"],
            "research_date": article["research_date"],
            "author": article["author"],
        }
        return Document(
            document_id=_doc_id(url),
            contents=contents,
            url=url,
            title=article["title"] or item["title"],
            summary=item["meta_info"],
            source_name=self.get_source_name(),
            locale="global",
            metadata=metadata,
        )

    def _document_contents(self, article):
        structured = []
        if article["statement"]:
            structured.append(f"PolitiFact statement: {article['statement']}")
        if article["verdict"]:
            structured.append(f"PolitiFact verdict: {article['verdict']}")
        if article["speaker"]:
            structured.append(f"PolitiFact speaker: {article['speaker']}")
        return [*structured, *article["paragraphs"]]


def _text(node, selector):
    tag = node.select_one(selector)
    return tag.get_text(strip=True) if tag else ""


def _list_verdict(container):
    img = container.select_one("div.m-result__media img")
    if img is None:
        return None
    return img.get("alt") or _verdict_from_src(img.get("src", ""))


def _article_verdict(soup):
    img = soup.select_one("img.c-image__original")
    if img and img.get("alt"):
        return img["alt"]
    meter_img = soup.select_one("div.m-statement__meter img")
    if meter_img and meter_img.get("alt"):
        return meter_img["alt"]
    return None


def _extract_paragraphs(soup):
    for selector in ("article.m-textblock", "div.short-on-time"):
        container = soup.select_one(selector)
        if container is None:
            continue
        paragraphs = clean_content_blocks([p.get_text(" ", strip=True) for p in container.find_all("p")])
        if paragraphs:
            return paragraphs
    return clean_content_blocks([p.get_text(" ", strip=True) for p in soup.find_all("p")], min_chars=50)


def _verdict_from_src(src):
    src = src.lower()
    if "pants" in src:
        return "Pants on Fire"
    if "false" in src:
        return "Mostly False" if "mostly" in src else "False"
    if "true" in src:
        return "Mostly True" if "mostly" in src else "True"
    if "half" in src:
        return "Half True"
    return None


def _doc_id(url):
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"politifact-{digest}"
