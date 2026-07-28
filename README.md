# GPE

This repository contains the code and datasets for running the GPE evaluation,
KnownGraph queries, and evidence retrieval.

## Data

- `gpe/data/gpe.jsonl`: the evaluation dataset and evidence corpus.
- `gpe/data/graph/`: the KnownGraph data, split into `claims.jsonl`,
  `entities.jsonl`, `events.jsonl`, and `sources.jsonl`.

## Setup

Install the package from the tagged GitHub source (the datasets are included):

```bash
python -m pip install "gpe-eval @ git+https://github.com/Aquarids/GPE-pub.git@v0.1.0"
```

To add GPE to another Conda environment, put the following entry under that
environment's `pip:` dependencies, then run `conda env update -f environment.yml`:

```yaml
- gpe-eval @ git+https://github.com/Aquarids/GPE-pub.git@v0.1.0
```

For development, the existing Conda environment remains available:

```bash
conda env create -f environment.yml
conda activate gpe
```

Copy the example environment file and add an API key:

```bash
cp .env.example .env
```

At minimum, set `LLM_API_KEY` in `.env`. The model and API endpoint can also be
changed with `LLM_MODEL` and `LLM_BASE_URL`.

## Graph and retrieval example

The graph and retrieval example reads only the included datasets and does not
require an API key. From the repository root, run:

```bash
python exp/run_graph_and_retrieval.py \
  --entity BBC \
  --claim-id benchmark-000001 \
  --query "steel production emissions" \
  --top-k 3
```

It prints KnownGraph statistics, matching entities, and the selected claim's
graph context. It then runs two internal retrieval examples over the same
query: a claim-local Top-K restricted to `--claim-id`, followed by a global
Top-K over all bundled evidence. Both examples include benign and poisoned
records, and each result includes its originating `claim_id`.

## Search interfaces

There are two distinct search APIs:

- `EvidenceRetriever.search(...)` retrieves from the bundled evidence corpus.
  It is global when `claim_id=None` and restricted to one claim when a
  `claim_id` is supplied. The experiment script shows both modes side by side.
  By default, its corpus includes `evidence_environment.related_distractor`
  records: entity-related but claim-irrelevant records whose `contents` is an
  empty list. To remove them for an ablation, pass
  `exclude_related_distractors=True` to `search(...)`. This option affects both
  BM25 candidate retrieval and LLM reranking.
  The same default and option apply to `EvidenceRequest` with retrieval source
  `"local"` or `"global"`.
  Select `mode="bm25"` for lexical retrieval or
  `mode="llm"` with an `LLMWrapper` argument for LLM reranking. The LLM mode
  first uses BM25 to recall candidates, then invokes the
  `rank_evidence_summaries` retrieval skill to rank only their titles,
  summaries, and keywords. It does not require or parse free-form JSON output,
  and it never sends full evidence contents to the LLM.
- `GPE.search(...)` uses `Searcher` to query external sources (for example,
  neutral web search, Wikipedia, or site-restricted search). The default
  `source="web"` submits the caller-provided `query` unchanged; select another
  source with the optional `source` argument.

## Retrieval evaluation protocols

`exp/` provides two matched internal-retrieval examples. Both use the current
claim as the retrieval query and pass only final Top-K records to the detector,
but poisoning is applied at a different stage in each protocol.

- `run_claim_retrieval_evaluation.py` retrieves Top-K only from the current
  `claim_id`'s benign evidence pool, then replaces `round(Top-K ×
  poison_ratio)` retrieved records with the configured attack. Thus poison does
  not affect local retrieval or reranking, and the final evidence list has the
  requested attack ratio.
- `run_global_retrieval_evaluation.py` first constructs evidence for all 638
  benchmark claims, combines those records into one pool for the condition,
  then retrieves Top-K from that global pool. The pool is prebuilt once before
  worker threads start and shared by all workers.

This pair separates claim-local retrieval from retrieval plus cross-claim
aggregation under the same evidence construction and attack condition. Each
result records the retrieval source, pool size, retrieval mode, candidate count,
and Top-K settings.

`EvidenceRequest.top_k` controls the final number of retrieved records passed
to a detector. Set `EvidenceRequest.pool_top_k` when the number of records each
claim contributes to the global pool should differ from that final Top-K; when
omitted, it defaults to `top_k` and its effective value is recorded in results.

`source="dataset"` remains available through `EvidenceRequest` as a controlled
claim-local baseline; it bypasses global retrieval.

## Evaluation

Run a one-example claim-local smoke test from the repository root:

```bash
python exp/run_claim_retrieval_evaluation.py \
  --method direct_evidence \
  --smoke-test \
  --threads 1
```

Run the matched global-pool smoke test:

```bash
python exp/run_global_retrieval_evaluation.py \
  --method direct_evidence \
  --smoke-test \
  --threads 1
```

Run the complete global-pool evaluation:

```bash
python exp/run_global_retrieval_evaluation.py \
  --method direct_evidence \
  --ratio 0 \
  --threads 5
```

Results are written separately to `output/ds/evaluation/local/` and
`output/ds/evaluation/global/`. Existing completed examples are skipped unless
`--overwrite` is supplied. The available methods are `direct_claim` and
`direct_evidence`.

## Token accounting

Evaluation results record LLM usage in two separate scopes. They must be
reported separately and are not added together.

- `overall_usage` / the summary's `overall.token_usage` covers the claim-level
  prediction. It starts before evidence resolution and ends after the detector
  returns the claim label, so it includes LLM retrieval reranking when
  `retrieval_mode="llm"`, as well as the final claim-verification call. BM25
  retrieval itself has no LLM token cost.
- Each entry in `subclaims[*].usage` / the summary's
  `subclaim.token_usage` covers one subclaim prediction. The summary sums these
  calls only; it does not charge the shared retrieval step again and does not
  include the claim-level prediction.

For each scope, `token_usage` reports provider-reported `prompt_tokens`,
`completion_tokens`, `total_tokens`, and `request_count`. The summary also
reports three token-efficiency measures, using exact-label correctness:

- `tcv` (tokens per correct verification) = `total_tokens / correct`;
- `correct_per_1k_tokens` = `1000 * correct / total_tokens`;
- `tokens_per_prediction` = `total_tokens / total_predictions`.

When no prediction is correct, `tcv` is reported as `null`. The per-result
`overall_usage.by_call_type` field additionally separates dialogue, generation,
and tool-call usage when supplied by the LLM provider.

## Build a GitHub release artifact

Build a source archive and wheel with the bundled datasets:

```bash
python -m pip install --upgrade build
python -m build
```

The artifacts are written to `dist/`. Update `version` in `pyproject.toml`,
create a matching Git tag and GitHub Release, then attach the files in `dist/`
to that release. No PyPI publication is required.
