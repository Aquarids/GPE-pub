import re
import time

import requests
from bs4 import BeautifulSoup

from gpe.retrieval.document import Document
from gpe.retrieval.search.ddg_base import DDGBaseSearch
from gpe.retrieval.search.errors import SearchProviderError


OEMBED_URL = "https://publish.twitter.com/oembed"
URL_PATTERN = re.compile(r"twitter\.com/([^/?#]+)/status/(\d+)")


class XSearch(DDGBaseSearch):
    NAMESPACE = "x"
    SOURCE_NAME = "x"
    DEFAULT_SUFFIX = ["site:x.com OR site:twitter.com"]

    def get_namespace(self):
        return self.NAMESPACE

    def get_source_name(self):
        return self.SOURCE_NAME

    def search(self, query, top_k=5, suffix=None, **kwargs) -> list[Document]:
        if suffix is None:
            suffix = self.DEFAULT_SUFFIX
        return super().search(query, top_k=top_k, suffix=suffix, **kwargs)

    def _search(self, query, top_k, suffix=None) -> list[Document]:
        suffix = list(suffix or [])
        full_query = self._build_query(query, suffix)
        hits = self.ddg.search(full_query, top_k=top_k)

        documents = []
        for rank, hit in enumerate(hits, start=1):
            try:
                tweet = self._fetch_tweet(hit["url"])
            except SearchProviderError as error:
                self._append_last_error(str(error))
                if self.extra.get("allow_snippet_fallback"):
                    documents.append(self._to_snippet_document(hit, rank, suffix, str(error)))
                continue
            if tweet is None:
                if self.extra.get("allow_snippet_fallback"):
                    documents.append(self._to_snippet_document(hit, rank, suffix, "not an embeddable X/Twitter status URL"))
                continue
            documents.append(self._to_document(hit, tweet, rank, suffix))
        return documents

    def _fetch_tweet(self, url):
        url = url.replace("x.com", "twitter.com")
        match = URL_PATTERN.search(url)
        if not match:
            return None

        time.sleep(self.extra.get("oembed_rate_limit", 1.5))

        try:
            response = requests.get(
                OEMBED_URL,
                params={"url": url, "omit_script": "1"},
                timeout=self.extra.get("oembed_timeout", 10),
                proxies=self.requests_proxies(),
            )
        except requests.RequestException as error:
            raise SearchProviderError("x", "oembed request failed", error, url=url) from error
        if response.status_code != 200:
            raise SearchProviderError("x", f"oembed returned HTTP {response.status_code}", url=url)

        try:
            payload = response.json()
        except ValueError as error:
            raise SearchProviderError("x", "oembed returned invalid JSON", error, url=url) from error
        paragraph = BeautifulSoup(payload.get("html", ""), "html.parser").find("p")
        text = paragraph.get_text(" ", strip=True) if paragraph else ""
        if not text:
            return None

        return {
            "text": text,
            "author_name": payload.get("author_name", ""),
            "author_url": payload.get("author_url", ""),
            "username": match.group(1),
            "tweet_id": match.group(2),
        }

    def _to_document(self, hit, tweet, rank, suffix) -> Document:
        url = hit["url"]
        metadata = {
            "rank": rank,
            "snippet": hit.get("snippet", ""),
            "author_name": tweet["author_name"],
            "author_url": tweet["author_url"],
            "username": tweet["username"],
            "tweet_id": tweet["tweet_id"],
            "query_suffix": list(suffix),
        }
        return Document(
            document_id=_doc_id(tweet["tweet_id"]),
            contents=[tweet["text"]],
            url=url,
            title=f"@{tweet['username']}",
            summary=hit.get("snippet", ""),
            source_name=self.get_source_name(),
            locale="global",
            metadata=metadata,
        )


def _doc_id(tweet_id):
    return f"x-{tweet_id}"
