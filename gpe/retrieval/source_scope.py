import hashlib
import re
from urllib.parse import urlparse


MAX_TOP_K = 10

SOURCE_DOMAINS = {
    "web": (),
    "politifact": ("politifact.com",),
    "wiki": ("wikipedia.org",),
    "academic": (
        "arxiv.org", "nature.com", "springer.com", "ncbi.nlm.nih.gov",
        "frontiersin.org", "researchgate.net", "tandfonline.com", "wiley.com",
        "sciencedirect.com", "jstor.org", "cambridge.org", "oup.com", "plos.org",
        "mdpi.com", "sagepub.com", "cell.com", "science.org", "nejm.org",
        "thelancet.com", "bmj.com", "acm.org", "ieee.org", "biorxiv.org",
        "medrxiv.org", "semanticscholar.org", "doi.org", "pnas.org",
        "royalsocietypublishing.org",
    ),
    "social": (
        "x.com", "twitter.com", "reddit.com", "facebook.com", "instagram.com",
        "threads.net", "tiktok.com", "youtube.com", "youtu.be", "linkedin.com",
        "bsky.app", "mastodon.social",
    ),
    "other": (),
}

SOURCE_MEMBERS = {
    "web": ("web", "politifact", "wiki", "academic", "x", "reddit", "facebook", "other"),
    "politifact": ("politifact",),
    "wiki": ("wiki",),
    "academic": ("arxiv", "academic"),
    "social": ("x", "reddit", "facebook", "instagram", "threads", "tiktok", "youtube", "linkedin"),
    "other": ("other",),
}

def list_source_scopes():
    return tuple(SOURCE_DOMAINS)


def source_members(source_scope):
    if source_scope not in SOURCE_MEMBERS:
        raise KeyError(f"unknown retrieval source scope: {source_scope}")
    return SOURCE_MEMBERS[source_scope]


def hostname(value):
    host = urlparse(str(value or "")).hostname or ""
    return host.casefold().removeprefix("www.")


def account_type(item):
    retrieval = item.get("retrieval") or {}
    metadata = retrieval.get("provider_metadata") or item.get("metadata") or {}
    value = metadata.get("account_type") or item.get("account_type")
    if value is None and metadata.get("official_account") is not None:
        value = "official" if metadata["official_account"] else "nonofficial"
    normalized = str(value or "unknown").strip().casefold().replace("-", "_")
    aliases = {
        "verified_official": "official",
        "organization": "official",
        "institution": "official",
        "unofficial": "nonofficial",
        "personal": "nonofficial",
        "individual": "nonofficial",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"official", "nonofficial", "community"} else "unknown"


def matches_source_scope(item, source_scope="web", include_poisoned=True):
    if source_scope not in SOURCE_DOMAINS:
        raise KeyError(f"unknown retrieval source scope: {source_scope}")
    if include_poisoned and item.get("evidence_type") == "poisoned":
        return True
    if source_scope == "web":
        return True
    return classify_source_scope(item) == source_scope


def classify_source_scope(item):
    retrieval_source = item.get("retrieval_source")
    if retrieval_source in {"x", "reddit", "facebook", "instagram", "threads", "tiktok", "youtube", "linkedin"}:
        return "social"
    if retrieval_source == "arxiv":
        return "academic"
    if retrieval_source in {"politifact", "wiki", "academic"}:
        return retrieval_source
    host = hostname(item.get("url"))
    for source_scope in ("politifact", "wiki", "academic", "social"):
        if any(
            host == domain or host.endswith(f".{domain}")
            for domain in SOURCE_DOMAINS[source_scope]
        ):
            return source_scope
    if host.endswith(".edu") or ".ac." in host:
        return "academic"
    return "other"


def source_provenance(item, source_scope="web"):
    existing = item.get("source_provenance")
    if isinstance(existing, dict):
        return dict(existing)
    poisoned = item.get("evidence_type") == "poisoned"
    profile = {"channel": source_scope, "authority_type": "unknown", "platform": None}
    if source_scope == "web":
        profile.update({
            "authority_type": "unverified" if poisoned else "domain_observed",
            "platform": hostname(item.get("url")) or None,
        })
    elif source_scope == "politifact":
        profile.update({
            "authority_type": "independent_fact_check" if poisoned else "official_platform",
            "platform": hostname(item.get("url")) or "politifact",
        })
    elif source_scope == "wiki":
        profile.update({
            "authority_type": "community_edited",
            "platform": "wikipedia",
        })
    elif source_scope == "academic":
        profile.update({
            "authority_type": "author_submission" if poisoned else "academic_source",
            "platform": hostname(item.get("url")) or item.get("retrieval_source") or "academic",
        })
    elif source_scope == "social":
        retrieval_source = item.get("retrieval_source")
        if not retrieval_source:
            item_host = hostname(item.get("url"))
            if item_host == "reddit.com" or item_host.endswith(".reddit.com"):
                retrieval_source = "reddit"
            elif item_host in {"x.com", "twitter.com"} or item_host.endswith(
                (".x.com", ".twitter.com")
            ):
                retrieval_source = "x"
            elif item_host == "facebook.com" or item_host.endswith(".facebook.com"):
                retrieval_source = "facebook"
            elif item_host == "instagram.com" or item_host.endswith(".instagram.com"):
                retrieval_source = "instagram"
            elif item_host == "youtube.com" or item_host.endswith(".youtube.com") or item_host == "youtu.be":
                retrieval_source = "youtube"
            elif item_host == "linkedin.com" or item_host.endswith(".linkedin.com"):
                retrieval_source = "linkedin"
        item_account_type = account_type(item)
        if poisoned:
            item_account_type = "nonofficial"
        elif retrieval_source == "reddit" and item_account_type == "unknown":
            item_account_type = "community"
        profile.update({
            "authority_type": item_account_type,
            "platform": retrieval_source or "social",
        })
    elif source_scope == "other":
        profile.update({
            "authority_type": "domain_observed",
            "platform": hostname(item.get("url")) or None,
        })
    return profile


def project_poisoned_item(item, source_scope="web"):
    projected = dict(item)
    if item.get("evidence_type") != "poisoned":
        return projected
    projected.pop("source_provenance", None)
    variant_scope = "web" if source_scope == "web" else source_scope
    variant = (item.get("publication_variants") or {}).get(variant_scope)
    if isinstance(variant, dict):
        projected.update(variant)
    return projected


def fixed_publication_variants(item, ordinal=None):
    """Build immutable, evidence-ID-derived publication metadata for attack records."""
    evidence_id = str(item.get("evidence_id") or "poisoned")
    digest = hashlib.sha1(evidence_id.encode("utf-8")).hexdigest()
    slug = re.sub(r"[^a-z0-9]+", "-", str(item.get("title") or evidence_id).casefold())
    slug = slug.strip("-")[:72] or digest[:12]
    outlet_names = (
        "Civic Bulletin",
        "Current Affairs Ledger",
        "Open Report Desk",
        "Public Record Monitor",
        "Regional News Review",
    )
    outlet_name = outlet_names[int(digest[:2], 16) % len(outlet_names)]
    base = {
        "url": item.get("url") or "",
        "source_name": outlet_name,
        "source_scope": "other",
    }
    fact_host = f"politifact-review-{digest[:10]}.net"
    publication_number = int(ordinal) if ordinal is not None else int(digest[:8], 16) % 99999 + 1
    wiki_revision = str(1300000000 + publication_number)
    arxiv_id = f"2609.{publication_number:05d}"
    social_platform = int(digest[14:16], 16) % 3
    if social_platform == 0:
        social = {
            "url": f"https://x.com/source_{digest[:8]}/status/{int(digest[:15], 16)}",
            "source_name": "x",
            "source_scope": "social",
            "retrieval_source": "x",
            "account_type": "nonofficial",
            "author": f"@source_{digest[:8]}",
        }
    elif social_platform == 1:
        social = {
            "url": f"https://www.reddit.com/r/currentevents/comments/{digest[:8]}/{slug}/",
            "source_name": "reddit",
            "source_scope": "social",
            "retrieval_source": "reddit",
            "account_type": "community",
            "author": f"u/source_{digest[:8]}",
        }
    else:
        social = {
            "url": f"https://www.facebook.com/source.{digest[:8]}/posts/{int(digest[:15], 16)}",
            "source_name": "facebook",
            "source_scope": "social",
            "retrieval_source": "facebook",
            "account_type": "nonofficial",
            "author": f"source.{digest[:8]}",
        }
    other = dict(base)
    return {
        "web": dict(base),
        "other": other,
        "politifact": {
            "url": f"https://{fact_host}/{slug}",
            "source_name": "PolitiFact Review Desk",
            "source_scope": "politifact",
            "retrieval_source": "politifact",
        },
        "wiki": {
            "url": f"https://en.wikipedia.org/w/index.php?title={slug}&oldid={wiki_revision}",
            "source_name": "wikipedia",
            "source_scope": "wiki",
            "retrieval_source": "wiki",
            "author": f"Editor-{digest[:8]}",
            "revision_id": wiki_revision,
        },
        "academic": {
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "source_name": "arxiv",
            "source_scope": "academic",
            "retrieval_source": "arxiv",
            "submission_id": arxiv_id,
        },
        "social": social,
    }


PUBLIC_EVIDENCE_FIELDS = {
    "title",
    "url",
    "summary",
    "contents",
    "source_name",
    "source_scope",
    "locale",
    "author",
    "published_at",
    "account_type",
    "revision_id",
    "submission_id",
    "source_provenance",
}


def public_evidence_view(item, source_scope="web"):
    projected = project_poisoned_item(item, source_scope)
    result = {
        key: projected.get(key)
        for key in PUBLIC_EVIDENCE_FIELDS
        if projected.get(key) is not None and projected.get(key) != ""
    }
    provenance = source_provenance(projected, source_scope)
    result["source_provenance"] = {
        key: value
        for key, value in provenance.items()
        if key in {
            "channel",
            "platform",
            "authority_type",
            "account_type",
            "revision_id",
            "submission_id",
        }
        and value not in {None, ""}
    }
    return result
