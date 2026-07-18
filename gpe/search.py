import os

from gpe.dataloader import ClaimLoader
from gpe.retrieval.search.arxiv_search import ArxivSearch
from gpe.retrieval.search.politifact_search import PolitifactSearch
from gpe.retrieval.search.reddit_search import RedditSearch
from gpe.retrieval.search.site_restricted_search import SiteRestrictedSearch
from gpe.retrieval.search.web_refute_search import WebRefuteSearch
from gpe.retrieval.search.web_support_search import WebSupportSearch
from gpe.retrieval.search.wiki_search import WikiSearch
from gpe.retrieval.search.x_search import XSearch


class Searcher:
    SEARCH_SOURCES = {
        "web_support": {"class": WebSupportSearch, "extra": {"ddg_region": "wt-wt", "ddg_safesearch": "moderate", "article_timeout": 10}},
        "web_refute": {"class": WebRefuteSearch, "extra": {"ddg_region": "wt-wt", "ddg_safesearch": "moderate", "article_timeout": 10}},
        "bbc": {"class": SiteRestrictedSearch, "extra": {"namespace": "bbc", "source_name": "BBC", "site_domains": ["bbc.com", "bbc.co.uk"], "source_region": "global", "source_reputation": 0.5, "article_timeout": 10}},
        "nbc": {"class": SiteRestrictedSearch, "extra": {"namespace": "nbc", "source_name": "NBC News", "site_domains": ["nbcnews.com"], "source_region": "us", "source_reputation": 0.5, "article_timeout": 10}},
        "cctv": {"class": SiteRestrictedSearch, "extra": {"namespace": "cctv", "source_name": "CCTV", "site_domains": ["cctv.com", "cgtn.com"], "source_region": "cn", "source_reputation": 0.5, "article_timeout": 10}},
        "xinhua": {"class": SiteRestrictedSearch, "extra": {"namespace": "xinhua", "source_name": "Xinhua", "site_domains": ["xinhuanet.com", "news.cn"], "source_region": "cn", "source_reputation": 0.5, "article_timeout": 10}},
        "piyao": {"class": SiteRestrictedSearch, "extra": {"namespace": "piyao", "source_name": "China Internet Joint Rumor Refutation Platform", "site_domains": ["piyao.org.cn"], "source_region": "cn", "source_reputation": 0.5, "article_timeout": 10}},
        "weibo": {"class": SiteRestrictedSearch, "extra": {"namespace": "weibo", "source_name": "Weibo", "site_domains": ["weibo.com", "m.weibo.cn"], "source_region": "cn", "source_reputation": 0.5, "article_timeout": 8, "allow_snippet_fallback": True}},
        "xiaohongshu": {"class": SiteRestrictedSearch, "extra": {"namespace": "xiaohongshu", "source_name": "Xiaohongshu", "site_domains": ["xiaohongshu.com", "xhslink.com"], "source_region": "cn", "source_reputation": 0.5, "article_timeout": 8, "allow_snippet_fallback": True}},
        "x": {"class": XSearch, "extra": {"ddg_region": "wt-wt", "ddg_safesearch": "moderate", "oembed_timeout": 10, "oembed_rate_limit": 1.5}},
        "politifact": {"class": PolitifactSearch, "extra": {"timeout": 15, "rate_limit": 1.5}},
        "wiki": {"class": WikiSearch, "extra": {"access_token": os.getenv("WIKI_ACCESS_TOKEN") or None, "language": os.getenv("WIKI_LANGUAGE", "en"), "user_agent": os.getenv("WIKI_USER_AGENT", "GeoPoisonEval/0.1"), "rate_limit": 0.72, "timeout": 15}},
        "arxiv": {"class": ArxivSearch, "extra": {"page_size": 100, "rate_limit": 3.0, "num_retries": 3}},
        "reddit": {"class": RedditSearch, "extra": {"timeout": 10, "rate_limit": 1.0, "include_comments": False, "max_comments": 10}},
    }

    def __init__(self, claim_loader=None, logger=None):
        self.claim_loader = claim_loader or ClaimLoader()
        self.logger = logger

    def search(self, query, top_k=5, source="web_support", extra=None):
        return self._build_search(source, extra).search(query, top_k=top_k)

    def search_by_claim_id(self, claim_id, top_k=5, source="web_support", extra=None):
        claim = self.claim_loader.get_claim(claim_id)
        query = claim.get("original_claim") or claim.get("claim") or ""
        return self.search(query, top_k, source, extra)

    def list_sources(self):
        return sorted(self.SEARCH_SOURCES)

    def _build_search(self, source, extra=None):
        if source not in self.SEARCH_SOURCES:
            raise KeyError(f"unknown search source: {source}")
        config = self.SEARCH_SOURCES[source]
        merged_extra = dict(config["extra"])
        merged_extra.update(extra or {})
        return config["class"](extra=merged_extra)
