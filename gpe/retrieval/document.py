class Document:
    def __init__(
        self,
        document_id=None,
        contents=None,
        url="",
        title=None,
        summary="",
        source_name=None,
        locale="global",
        score=None,
        credibility=None,
        metadata=None,
        extra=None,
    ):
        base_contents = list(contents or [])

        self.document_id = document_id
        self.contents = [str(item) for item in base_contents if str(item).strip()]
        self.url = str(url or "")
        self.title = title
        self.summary = summary
        self.source_name = source_name
        self.locale = locale
        self.score = score
        self.credibility = credibility
        self.metadata = dict(metadata or {})
        self.extra = dict(extra or {})

    @property
    def text(self):
        return "\n".join(self.contents)

    @property
    def content_count(self):
        return len(self.contents)

    def to_dict(self):
        return {
            "document_id": self.document_id,
            "contents": list(self.contents),
            "url": self.url,
            "title": self.title,
            "summary": self.summary,
            "source_name": self.source_name,
            "locale": self.locale,
            "score": self.score,
            "credibility": self.credibility,
            "metadata": dict(self.metadata),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data))
