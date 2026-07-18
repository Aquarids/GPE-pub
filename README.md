# GPE

This repository contains the code and datasets for running the GPE evaluation,
KnownGraph queries, and evidence retrieval.

## Data

- `gpe/data/gpe.jsonl`: the evaluation dataset and evidence corpus.
- `gpe/data/graph/`: the KnownGraph data, split into `claims.jsonl`,
  `entities.jsonl`, `events.jsonl`, and `sources.jsonl`.

## Setup

Install the package (the datasets are included in the installation):

```bash
python -m pip install .
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
require an API key. After installation, run:

```bash
gpe-graph \
  --entity BBC \
  --claim-id benchmark-000001 \
  --query "steel production emissions" \
  --top-k 3
```

It prints KnownGraph statistics, matching entities, the selected claim's graph
context, and the top matching benign evidence records. To include poisoned
evidence in the retrieval results, add `--include-poisoned`.

## Evaluation

Run a one-example smoke test from the repository root:

```bash
gpe-evaluate \
  --method direct_evidence \
  --smoke-test \
  --threads 1
```

Run the clean evaluation on the complete dataset:

```bash
gpe-evaluate \
  --method direct_evidence \
  --ratio 0 \
  --threads 5
```

Results are written to `output/.../evaluation/`. Existing completed examples are
skipped unless `--overwrite` is supplied.

## Build and publish

Build a source archive and wheel with the bundled datasets:

```bash
python -m pip install --upgrade build
python -m build
```

The artifacts are written to `dist/`. Before publishing, change `name` and
`version` in `pyproject.toml` as appropriate, then upload with your chosen
package registry workflow.
