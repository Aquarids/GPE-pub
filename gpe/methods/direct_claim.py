from typing import Any, Dict

from gpe.methods.base import BaseDetector
from gpe.methods.label_utils import normalize_compare_label
from gpe.utils.json_utils import extract_json


class DirectClaimDetector(BaseDetector):
    """Judge a claim from model-parametric knowledge without external evidence."""

    SYSTEM_PROMPT = """You are a fact-verification system.
Judge the input claim directly using your internal knowledge. Return JSON only:
{
  "verdict": "true|mostly_true|half_true|mostly_false|false|uncertain",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}
Use uncertain only when the claim cannot be reliably determined.
"""

    def detect(self, claim: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        response = self.llm.dialogue(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Claim:\n{claim}"},
            ],
            stream=True,
        )
        payload = extract_json(response)
        verdict = str(payload.get("verdict") or "uncertain").strip().lower().replace("-", "_")
        if verdict != "uncertain":
            verdict = normalize_compare_label(verdict)
        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": max(0.0, min(1.0, float(payload.get("confidence", 0.0)))),
            "explanation": str(payload.get("reasoning") or "").strip(),
            "method": "DirectClaim",
        }
