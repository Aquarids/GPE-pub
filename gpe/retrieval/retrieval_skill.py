"""Tool-backed skill for selecting relevant evidence summaries."""

import json
import os


TOOL_NAME = "rank_evidence_summaries"

EVIDENCE_RANKING_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Select and rank the evidence records most relevant to a query. "
            "Use only record IDs provided in the candidate list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "record_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Candidate record IDs in descending relevance order.",
                },
            },
            "required": ["record_ids"],
            "additionalProperties": False,
        },
    },
}


def rank_evidence_summaries(llm, query, candidates, top_k):
    """Use a tool call to rank candidate summaries without free-form JSON output."""
    if not hasattr(llm, "call_function"):
        raise TypeError("LLM ranking requires an LLM wrapper with call_function()")

    payload = {
        "query": query,
        "candidates": [
            {
                "record_id": item["record_id"],
                "title": item.get("title"),
                "summary": item.get("summary"),
                "keywords": (item.get("retrieval") or {}).get("keywords", []),
            }
            for item in candidates
        ],
        "top_k": top_k,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Rank evidence by relevance to the query. Base the decision only on "
                "the supplied titles, summaries, and keywords. Call the ranking tool "
                "exactly once; do not produce a text response."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    attempts = max(1, int(os.getenv("LLM_TOOL_CALL_ATTEMPTS", "3")))
    matching = []
    for _ in range(attempts):
        calls = llm.call_function(
            messages,
            [EVIDENCE_RANKING_TOOL],
            tool_choice=os.getenv("LLM_TOOL_CHOICE", "auto"),
        )
        matching = [call for call in calls if call.get("name") == TOOL_NAME]
        if matching:
            break
        messages.append(
            {
                "role": "user",
                "content": "Call rank_evidence_summaries now. Do not reply with text.",
            }
        )
    if len(matching) != 1:
        raise RuntimeError(
            f"expected one {TOOL_NAME} call after {attempts} attempts, received {len(matching)}"
        )
    arguments = matching[0].get("arguments") or "{}"
    values = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    record_ids = values.get("record_ids")
    if not isinstance(record_ids, list) or not all(isinstance(item, str) for item in record_ids):
        raise ValueError(f"{TOOL_NAME} must provide record_ids as a list of strings")
    return record_ids
