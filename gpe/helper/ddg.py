import logging

try:
    from ddgs import DDGS
except ModuleNotFoundError as error:
    DDGS = None
    DDGS_IMPORT_ERROR = error
else:
    DDGS_IMPORT_ERROR = None

from gpe.helper.errors import SearchProviderError


LOGGER = logging.getLogger(__name__)


class DDG:
    def __init__(self, region="wt-wt", safesearch="moderate", proxy=None, backend="duckduckgo,brave,yahoo"):
        self.region = region
        self.safesearch = safesearch
        self.proxy = proxy
        self.backend = backend
        self.last_error = ""

    def search(self, query, top_k=5):
        self.last_error = ""
        if DDGS is None:
            self.last_error = "DDG search failed: ddgs dependency is not installed"
            raise SearchProviderError("ddg", "ddgs dependency is not installed", DDGS_IMPORT_ERROR, query=query)

        try:
            kwargs = {"proxy": self.proxy} if self.proxy else {}
            with DDGS(**kwargs) as ddgs:
                raw = ddgs.text(
                    query,
                    region=self.region,
                    safesearch=self.safesearch,
                    max_results=top_k,
                    backend=self.backend,
                )
        except Exception as error:
            self.last_error = f"DDG search failed: {type(error).__name__}: {error}"
            LOGGER.warning("%s query=%r", self.last_error, query)
            raise SearchProviderError("ddg", "search request failed", error, query=query) from error

        hits = []
        for item in list(raw or []):
            url = item.get("href") or item.get("url")
            if not url:
                continue
            hits.append(
                {
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": item.get("body", "") or item.get("snippet", ""),
                }
            )
        return hits
