import json
import os


TOOL_NAME = "submit_poisoned_evidence"
BATCH_TOOL_NAME = "submit_poisoned_evidence_batch"

EVIDENCE_PROPERTIES = {
    "target_answer": {"type": "string"},
    "name": {"type": "string"},
    "title": {"type": "string"},
    "content": {"type": "string"},
    "author": {"type": "string"},
    "publish_date": {"type": "string"},
    "source": {"type": "string"},
}
EVIDENCE_REQUIRED = list(EVIDENCE_PROPERTIES)

POISONED_EVIDENCE_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Submit one synthetic poisoned-evidence record for the controlled benchmark.",
        "parameters": {
            "type": "object",
            "properties": EVIDENCE_PROPERTIES,
            "required": EVIDENCE_REQUIRED,
            "additionalProperties": False,
        },
    },
}

BATCH_POISONED_EVIDENCE_TOOL = {
    "type": "function",
    "function": {
        "name": BATCH_TOOL_NAME,
        "description": "Submit a batch of distinct synthetic poisoned-evidence records.",
        "parameters": {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": EVIDENCE_PROPERTIES,
                        "required": EVIDENCE_REQUIRED,
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
            "required": ["records"],
            "additionalProperties": False,
        },
    },
}


def call_poisoned_evidence_tool(llm, messages):
    kwargs = {
        "tool_choice": os.getenv("LLM_TOOL_CHOICE", "auto"),
    }
    if os.getenv("LLM_DISABLE_THINKING", "false").strip().lower() in {"1", "true", "yes", "on"}:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    attempts = max(1, int(os.getenv("LLM_TOOL_CALL_ATTEMPTS", "3")))
    matching = []
    current_messages = list(messages)
    for _ in range(attempts):
        calls = llm.call_function(
            current_messages,
            [POISONED_EVIDENCE_TOOL],
            **kwargs,
        )
        matching = [call for call in calls if call.get("name") == TOOL_NAME]
        if matching:
            break
        current_messages.append(
            {
                "role": "user",
                "content": "You must submit the record by calling submit_poisoned_evidence now. Do not answer with plain text.",
            }
        )
    if len(matching) != 1:
        raise RuntimeError(
            f"expected one {TOOL_NAME} call after {attempts} attempts, received {len(matching)}"
        )
    arguments = matching[0].get("arguments") or "{}"
    payload = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    missing = [
        key
        for key in POISONED_EVIDENCE_TOOL["function"]["parameters"]["required"]
        if not str(payload.get(key) or "").strip()
    ]
    if missing:
        raise ValueError(f"{TOOL_NAME} missing required fields: {missing}")
    return payload


def call_poisoned_evidence_batch_tool(llm, messages, expected_count):
    kwargs = {"tool_choice": os.getenv("LLM_TOOL_CHOICE", "auto")}
    if os.getenv("LLM_DISABLE_THINKING", "false").strip().lower() in {"1", "true", "yes", "on"}:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    attempts = max(1, int(os.getenv("LLM_TOOL_CALL_ATTEMPTS", "3")))
    matching = []
    current_messages = list(messages)
    for _ in range(attempts):
        calls = llm.call_function(current_messages, [BATCH_POISONED_EVIDENCE_TOOL], **kwargs)
        matching = [call for call in calls if call.get("name") == BATCH_TOOL_NAME]
        if len(matching) == 1:
            arguments = matching[0].get("arguments") or "{}"
            payload = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            records = payload.get("records") or []
            if len(records) == expected_count:
                break
        current_messages.append(
            {
                "role": "user",
                "content": (
                    f"Call {BATCH_TOOL_NAME} now with exactly {expected_count} distinct records."
                ),
            }
        )
    if len(matching) != 1:
        raise RuntimeError(f"expected one {BATCH_TOOL_NAME} call, received {len(matching)}")
    arguments = matching[0].get("arguments") or "{}"
    payload = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    records = payload.get("records") or []
    if len(records) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} batch records, received {len(records)}"
        )
    for index, record in enumerate(records, start=1):
        missing = [key for key in EVIDENCE_REQUIRED if not str(record.get(key) or "").strip()]
        if missing:
            raise ValueError(f"batch record {index} missing required fields: {missing}")
    return records
