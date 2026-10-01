#!/usr/bin/env python3
import argparse
import configparser
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

from gpe.exp import EvaluationCase, EvaluationPipeline, EvidenceRequest, GlobalEvidencePool, JsonlResultSink
from gpe.gpe import GPE
from gpe.helper.llm_wrapper import LLMWrapper
from gpe.helper.logger import ENV_LOCAL, Logger
from gpe.methods import DETECTORS
from gpe.retrieval.source_scope import MAX_TOP_K, list_source_scopes
from gpe.resources import GPE_DATA_PATH


DATA_PATH = GPE_DATA_PATH
POISON_DIR = None
DYNAMIC_POISON_DATA_PATH = {
    "fakegpt": "output/dynamic_poison/malicious_fakegpt.jsonl",
    "poisonedrag": "output/dynamic_poison/malicious_poisonedrag.jsonl",
    "ata": "output/dynamic_poison/malicious_ata.jsonl",
    "ignore": "output/dynamic_poison/malicious_ignore.jsonl",
}
OUTPUT_DIR = Path("output/evaluation")

METHODS = sorted(DETECTORS)
ATTACK_TYPES = ["fakegpt", "poisonedrag", "ata", "ignore"]
EVIDENCE_SOURCE = "dataset"
RETRIEVAL_MODE = "bm25"
RETRIEVAL_DATA_PATH = None
RETRIEVAL_CANDIDATE_K = 30
FILTER_BENIGN = False
POOL_TOP_K = None
RETRIEVAL_SOURCE = "web"
INCLUDE_DISTRACTORS = False

CATEGORY = None
LIMIT = None
TOP_K = 3
SEED = 0
INCLUDE_SUBCLAIMS = True
GENERATE_MISSING_POISON = False
QUIET = False
STREAM = True

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
API_TYPE = os.getenv("LLM_API_TYPE", "chat_completions")
RETRIES = 1
THREADS = 5

DETECTOR_CONFIG = {}


def parse_args(default_evidence_source=EVIDENCE_SOURCE):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()
    values = load_experiment_config(config_args.config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--data-path", default=values.get("data_path", DATA_PATH))
    parser.add_argument("--poison-dir", type=Path, default=values.get("poison_dir", POISON_DIR))
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--ratio", default="0")
    parser.add_argument("--output-dir", type=Path, default=values.get("output_dir", OUTPUT_DIR))
    parser.add_argument("--threads", type=int, default=config_int(values, "threads", THREADS))
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--attacks", default=None)
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-missing-usage", action="store_true")
    parser.add_argument("--no-subclaims", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--generate-missing-poison",
        action="store_true",
        default=config_bool(values, "generate_missing_poison", GENERATE_MISSING_POISON),
        help="Generate and cache missing poison records with the configured LLM.",
    )
    parser.add_argument("--evidence-source", choices=["dataset", "global", "search"], default=values.get("evidence_source", default_evidence_source))
    parser.add_argument("--retrieval-mode", choices=["bm25", "llm"], default=values.get("retrieval_mode", RETRIEVAL_MODE))
    parser.add_argument("--retrieval-data-path", default=values.get("retrieval_data_path", RETRIEVAL_DATA_PATH))
    parser.add_argument("--retrieval-candidate-k", type=int, default=config_int(values, "retrieval_candidate_k", RETRIEVAL_CANDIDATE_K))
    parser.add_argument("--top-k", type=int, default=config_int(values, "top_k", TOP_K))
    parser.add_argument("--pool-top-k", type=int, default=config_int(values, "pool_top_k", POOL_TOP_K))
    parser.add_argument("--filter-benign", action="store_true", default=config_bool(values, "filter_benign", FILTER_BENIGN))
    parser.add_argument(
        "--retrieval-source",
        choices=list_source_scopes(),
        default=values.get("retrieval_source", RETRIEVAL_SOURCE),
    )
    parser.add_argument(
        "--include-distractors",
        action="store_true",
        default=config_bool(values, "include_distractors", INCLUDE_DISTRACTORS),
    )
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=config_bool(values, "stream", STREAM))
    return parser.parse_args()


def load_experiment_config(path):
    if path is None:
        return {}
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(f"experiment config not found: {path}")
    if "experiment" not in parser:
        raise ValueError(f"missing [experiment] section in {path}")
    return dict(parser["experiment"])


def config_int(values, name, default):
    value = values.get(name)
    return default if value in {None, ""} else int(value)


def config_bool(values, name, default):
    value = values.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


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


def main(default_evidence_source=EVIDENCE_SOURCE):
    args = parse_args(default_evidence_source)
    if args.threads < 1:
        raise ValueError("threads must be at least 1")
    if not 1 <= args.top_k <= MAX_TOP_K:
        raise ValueError(f"top-k must be between 1 and {MAX_TOP_K}")
    if args.pool_top_k is not None and not 1 <= args.pool_top_k <= MAX_TOP_K:
        raise ValueError(f"pool-top-k must be between 1 and {MAX_TOP_K}")
    ratio = 0.0 if args.smoke_test else parse_ratio(args.ratio)
    attacks = parse_attacks(args.attacks)
    limit = 1 if args.smoke_test else args.limit
    include_subclaims = False if args.smoke_test else not args.no_subclaims
    output_root = args.output_dir / args.retrieval_mode if args.evidence_source == "global" else args.output_dir
    if args.evidence_source == "global" and (
        args.retrieval_source != "web" or args.include_distractors
    ):
        output_root = output_root / f"source_{args.retrieval_source}"
        if args.include_distractors:
            output_root = output_root / "with_distractors"
        output_root = output_root / f"top_k_{args.top_k}"
    elif args.top_k != TOP_K:
        output_root = output_root / f"top_k_{args.top_k}"
    output_dir = output_root / "smoke" if args.smoke_test else output_root / ratio_name(ratio)
    output_path = output_dir / f"{args.method}.jsonl"

    logger = Logger(f"gpe-evaluation-{args.method}-{args.evidence_source}-{ratio_name(ratio)}", ENV_LOCAL, enabled=not QUIET)
    llm = LLMWrapper(
        logger,
        model_id=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        api_type=API_TYPE,
        stream=args.stream,
        retries=RETRIES,
    )
    poison_paths = None
    if args.poison_dir is not None:
        poison_paths = {
            attack: args.poison_dir / f"malicious_{attack}.jsonl"
            for attack in ATTACK_TYPES
        }
    framework = GPE(
        data_path=args.data_path,
        poison_path=poison_paths,
        dynamic_poison_path=DYNAMIC_POISON_DATA_PATH,
        llm=llm,
        logger=logger,
    )
    sink = JsonlResultSink(
        output_path,
        overwrite=args.overwrite,
        rerun_missing_usage=args.rerun_missing_usage,
    )
    method_classes = dict(DETECTORS)
    config = dict(DETECTOR_CONFIG)
    config["evidence_source"] = "search" if args.evidence_source == "search" else "dataset"
    claims = framework.claims.list_claims(category=args.category)
    if limit is not None:
        claims = claims[:limit]
    global_pool = GlobalEvidencePool(
        framework.claims,
        framework.evidence,
        framework.poison_cache,
        claim_ids=None,
    )

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
                global_pool,
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
            source=args.evidence_source,
            top_k=args.top_k,
            poison_ratio=ratio,
            attack_type=attack_type,
            seed=SEED,
            generate_missing_poison=args.generate_missing_poison,
            retrieval_mode=args.retrieval_mode,
            retrieval_data_path=args.retrieval_data_path,
            retrieval_candidate_k=args.retrieval_candidate_k,
            filter_benign=args.filter_benign,
            pool_top_k=args.pool_top_k,
            retrieval_source=args.retrieval_source,
            include_distractors=args.include_distractors,
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
                "rerun_missing_usage": args.rerun_missing_usage,
                "smoke_test": args.smoke_test,
                "top_k": args.top_k,
                "claim_count": len(claims),
                "poison_dir": str(args.poison_dir) if args.poison_dir is not None else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
