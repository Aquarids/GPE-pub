import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpe.graph import GPEKnowledgeGraph
from gpe.retrieval import EvidenceRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="BBC")
    parser.add_argument("--query", default="steel production emissions")
    parser.add_argument("--claim-id", default="benchmark-000001")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--include-poisoned", action="store_true")
    args = parser.parse_args()

    graph = GPEKnowledgeGraph()
    print(json.dumps(graph.statistics(), ensure_ascii=False, indent=2))
    print(json.dumps(graph.find_entities(args.entity, limit=args.top_k), ensure_ascii=False, indent=2))
    print(json.dumps(graph.claim_context(args.claim_id), ensure_ascii=False, indent=2))

    retriever = EvidenceRetriever()
    results = retriever.search(
        args.query,
        top_k=args.top_k,
        filter_benign=not args.include_poisoned,
    )
    print(json.dumps([
        {
            "record_id": item["record_id"],
            "score": item["score"],
            "title": item.get("title"),
            "url": item.get("url"),
            "evidence_type": item.get("evidence_type"),
        }
        for item in results
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
