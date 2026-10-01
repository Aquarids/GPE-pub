__version__ = "0.2.1"

__all__ = [
    "__version__",
    "GPE",
    "EvaluationCase",
    "EvaluationPipeline",
    "EvidenceRequest",
    "JsonlResultSink",
    "GPEKnowledgeGraph",
    "EvidenceRetriever",
    "ClaimLoader",
    "DatasetEvidenceLoader",
    "list_source_scopes",
    "public_evidence_view",
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
    if name in {"ClaimLoader", "DatasetEvidenceLoader"}:
        from gpe.dataloader import ClaimLoader, DatasetEvidenceLoader
        return {"ClaimLoader": ClaimLoader, "DatasetEvidenceLoader": DatasetEvidenceLoader}[name]
    if name in {"list_source_scopes", "public_evidence_view"}:
        from gpe.retrieval.source_scope import list_source_scopes, public_evidence_view
        return {
            "list_source_scopes": list_source_scopes,
            "public_evidence_view": public_evidence_view,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
