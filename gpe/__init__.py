__all__ = [
    "GPE",
    "EvaluationCase",
    "EvaluationPipeline",
    "EvidenceRequest",
    "JsonlResultSink",
    "GPEKnowledgeGraph",
    "EvidenceRetriever",
]


def __getattr__(name):
    """Import public APIs only when they are requested.

    In particular, graph and retrieval users should not need optional LLM
    dependencies merely by invoking a lightweight command-line tool.
    """
    if name == "GPE":
        from gpe.gpe import GPE
        return GPE
    if name in {"EvaluationCase", "EvaluationPipeline", "EvidenceRequest", "JsonlResultSink"}:
        from gpe.exp import EvaluationCase, EvaluationPipeline, EvidenceRequest, JsonlResultSink
        return {
            "EvaluationCase": EvaluationCase,
            "EvaluationPipeline": EvaluationPipeline,
            "EvidenceRequest": EvidenceRequest,
            "JsonlResultSink": JsonlResultSink,
        }[name]
    if name == "GPEKnowledgeGraph":
        from gpe.graph import GPEKnowledgeGraph
        return GPEKnowledgeGraph
    if name == "EvidenceRetriever":
        from gpe.retrieval.evidence_retrieval import EvidenceRetriever
        return EvidenceRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
