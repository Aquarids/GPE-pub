from gpe.methods.direct_claim import DirectClaimDetector
from gpe.methods.direct_evidence import DirectEvidenceDetector


DETECTORS = {
    "direct_claim": DirectClaimDetector,
    "direct_evidence": DirectEvidenceDetector,
}

__all__ = ["DETECTORS"]
