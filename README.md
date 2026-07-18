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

It prints KnownGraph statistics, matching entities, the selected claim's graph
context, and the top matching evidence records. Retrieval ranks the full
corpus, including evidence attached to other claims and poisoned evidence, to
better model a real retrieval setting.

## Search interfaces

There are two distinct search APIs:

- `EvidenceRetriever.search(...)` retrieves from the bundled evidence corpus.
  It is global when `claim_id=None` and restricted to one claim when a
  `claim_id` is supplied. Select `mode="bm25"` for lexical retrieval or
  `mode="llm"` with an `LLMWrapper` argument for LLM reranking. The LLM mode
  first uses BM25 to recall candidates, then invokes the
  `rank_evidence_summaries` retrieval skill to rank only their titles,
  summaries, and keywords. It does not require or parse free-form JSON output,
  and it never sends full evidence contents to the LLM.
- `GPE.search(...)` uses `Searcher` to query external sources (for example,
  neutral web search, Wikipedia, or site-restricted search). The default
  `source="web"` submits the caller-provided `query` unchanged; select another
  source with the optional `source` argument.

The evaluation script's default `dataset` mode is different: it directly uses
the evidence pre-associated with each claim to provide a controlled evaluation
condition; it does not run corpus retrieval.

## Evaluation

Run a one-example smoke test from the repository root:

```bash
python exp/run_gpe_evaluation.py \
  --method direct_evidence \
  --smoke-test \
  --threads 1
```

Run the clean evaluation on the complete dataset:

```bash
python exp/run_gpe_evaluation.py \
  --method direct_evidence \
  --ratio 0 \
  --threads 5
```

Results are written to `output/.../evaluation/`. Existing completed examples are
skipped unless `--overwrite` is supplied.

## Build a GitHub release artifact

Build a source archive and wheel with the bundled datasets:

```bash
python -m pip install --upgrade build
python -m build
```

The artifacts are written to `dist/`. Update `version` in `pyproject.toml`,
create a matching Git tag and GitHub Release, then attach the files in `dist/`
to that release. No PyPI publication is required.
