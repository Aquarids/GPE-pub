#!/usr/bin/env python3
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import time

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from gpe.exp import EvaluationCase, EvaluationPipeline, EvidenceRequest, JsonlResultSink
from gpe.gpe import GPE
from gpe.helper.llm_wrapper import LLMWrapper
from gpe.helper.logger import ENV_LOCAL, Logger
from gpe.methods import DETECTORS
from gpe.resources import GPE_DATA_PATH


DATA_PATH = GPE_DATA_PATH
DYNAMIC_POISON_DATA_PATH = {
    "fakegpt": "output/dynamic_poison/malicious_fakegpt.jsonl",
    "poisonedrag": "output/dynamic_poison/malicious_poisonedrag.jsonl",
    "ata": "output/dynamic_poison/malicious_ata.jsonl",
    "ignore": "output/dynamic_poison/malicious_ignore.jsonl",
}
OUTPUT_DIR = Path("output/ds/evaluation")

METHODS = ["direct_evidence", "rafts", "safe", "steel"]
ATTACK_TYPES = ["fakegpt", "poisonedrag", "ata", "ignore"]
EVIDENCE_SOURCE = "dataset"

CATEGORY = None
LIMIT = None
TOP_K = 3
SEED = 0
INCLUDE_SUBCLAIMS = True
GENERATE_MISSING_POISON = False
QUIET = False

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
API_TYPE = os.getenv("LLM_API_TYPE", "chat_completions")
RETRIES = 1
THREADS = 5

DETECTOR_CONFIG = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--ratio", default="0")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--attacks", default=None)
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-subclaims", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def parse_ratio(value):
    value = str(value).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        ratio = float(numerator) / float(denominator)
    else:
        ratio = float(value)
    if ratio < 0 or ratio > 1:
        raise ValueError(f"invalid poison ratio: {value}")
    return ratio


def ratio_name(value):
    ratio = Fraction(value).limit_denominator(1000)
    if ratio.denominator == 1:
        return f"ratio_{ratio.numerator}"
    return f"ratio_{ratio.numerator}_{ratio.denominator}"


def parse_attacks(value):
    if not value:
        return list(ATTACK_TYPES)
    attacks = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(attacks) - set(ATTACK_TYPES))
    if unknown:
        raise ValueError(f"unknown attacks: {unknown}")
    return attacks


def main():
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be at least 1")
    ratio = 0.0 if args.smoke_test else parse_ratio(args.ratio)
    attacks = parse_attacks(args.attacks)
    limit = 1 if args.smoke_test else args.limit
    include_subclaims = False if args.smoke_test else not args.no_subclaims
    output_dir = args.output_dir / "smoke" if args.smoke_test else args.output_dir / ratio_name(ratio)
    output_path = output_dir / f"{args.method}.jsonl"

    logger = Logger(f"gpe-evaluation-{args.method}-{ratio_name(ratio)}", ENV_LOCAL, enabled=not QUIET)
    llm = LLMWrapper(
        logger,
        model_id=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        api_type=API_TYPE,
        stream=True,
        retries=RETRIES,
    )
    framework = GPE(
        data_path=DATA_PATH,
        dynamic_poison_path=DYNAMIC_POISON_DATA_PATH,
        llm=llm,
        logger=logger,
    )
    sink = JsonlResultSink(output_path, overwrite=args.overwrite)
    method_classes = {**DETECTORS}
    config = dict(DETECTOR_CONFIG)
    config["evidence_source"] = EVIDENCE_SOURCE
    claims = framework.claims.list_claims(category=args.category)
    if limit is not None:
        claims = claims[:limit]

    worker_state = threading.local()

    def worker_context():
        context = getattr(worker_state, "context", None)
        if context is None:
            worker_llm = llm.clone_for_worker()
            detector = method_classes[args.method](logger, worker_llm, config)
            pipeline = EvaluationPipeline(
                framework.claims,
                framework.evidence,
                framework.evaluator,
                worker_llm,
                framework.poison_cache,
            )
            context = (pipeline, detector)
            worker_state.context = context
        return context

    def evaluate_case(template):
        pipeline, detector = worker_context()
        case = EvaluationCase(
            method=template.method,
            detector=detector,
            claim_id=template.claim_id,
            evidence=template.evidence,
            include_subclaims=template.include_subclaims,
        )
        row = pipeline.evaluate(case)
        row["model_id"] = MODEL
        return sink.append(case, row)

    pending_cases = []
    ratio_attacks = [None] if ratio == 0 else attacks
    for attack_type in ratio_attacks:
        request = EvidenceRequest(
            source=EVIDENCE_SOURCE,
            top_k=TOP_K,
            poison_ratio=ratio,
            attack_type=attack_type,
            seed=SEED,
            generate_missing_poison=GENERATE_MISSING_POISON,
        )
        for claim in claims:
            case = EvaluationCase(
                method=args.method,
                detector=None,
                claim_id=claim["claim_id"],
                evidence=request,
                include_subclaims=include_subclaims,
            )
            if sink.contains(case):
                continue
            pending_cases.append(case)

    written = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {}
        for case in pending_cases:
            future = executor.submit(evaluate_case, case)
            futures[future] = case
            time.sleep(1)

        for future in as_completed(futures):

            case = futures[future]
            attack_type = case.evidence.attack_type
            try:
                appended = future.result()
            except Exception as error:
                logger.log_exception(error)
                print(
                    f"{args.method} ratio={ratio} attack={attack_type or 'clean'} "
                    f"claim={case.claim_id} status=error error={type(error).__name__}: {error}"
                )
                continue
            written += int(appended)
            print(
                f"{args.method} ratio={ratio} attack={attack_type or 'clean'} "
                f"claim={case.claim_id} status={'done' if appended else 'skipped'}"
            )

    print(
        json.dumps(
            {
                "method": args.method,
                "poison_ratio": ratio,
                "output_path": str(output_path),
                "written": written,
                "completed": len(sink.completed),
                "threads": args.threads,
                "smoke_test": args.smoke_test,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
