try:
    from newspaper import Article as NewsArticle
    from newspaper import Config
except ModuleNotFoundError as error:
    NewsArticle = None
    Config = None
    NEWSPAPER_IMPORT_ERROR = error
else:
    NEWSPAPER_IMPORT_ERROR = None

from gpe.helper.content_filter import clean_content_blocks
from gpe.helper.errors import SearchProviderError


class Article:
    def __init__(self, timeout=10, language=None, proxy=None):
        self.timeout = timeout
        self.language = language
        self.proxy = proxy
        self._config = self._build_config()

    def fetch(self, url):
        if NewsArticle is None:
            raise SearchProviderError(
                "article",
                "newspaper dependency is not installed",
                NEWSPAPER_IMPORT_ERROR,
                url=url,
            )

        article = NewsArticle(url, config=self._config)
        try:
            article.download()
            article.parse()
        except Exception as error:
            raise SearchProviderError("article", "fetch request failed", error, url=url) from error

        text = (article.text or "").strip()
        if not text:
            return None

        publish_date = article.publish_date
        return {
            "url": url,
            "title": article.title or "",
            "paragraphs": clean_content_blocks(text.split("\n")),
            "authors": list(article.authors or []),
            "publish_date": publish_date.isoformat() if publish_date else None,
            "top_image": article.top_image or "",
        }

    def _build_config(self):
        if Config is None:
            return None

        config = Config()
        config.request_timeout = self.timeout
        config.fetch_images = False
        config.memoize_articles = False
        if self.proxy:
            config.proxies = {"http": self.proxy, "https": self.proxy}
        if self.language:
            config.language = self.language
        return config
