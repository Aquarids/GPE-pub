# GPE

GPE is an evaluation framework and benchmark for measuring how GEO-style evidence poisoning affects systems that retrieve and judge Web evidence.

This public repository contains:

- the released GPE JSONL benchmark and JSON Schema;
- claim-local and global-pool evaluation;
- BM25 and LLM evidence retrieval;
- source-aware retrieval over Web, PolitiFact, Wikipedia, academic, social, and other evidence;
- FakeGPT, PoisonedRAG, ATA, and Ignore Injection attack implementations for custom evaluation data;
- public evidence projection that hides evaluator-only attack annotations from tested methods;
- live Web, PolitiFact, Wikipedia, arXiv, X, Reddit, and Facebook search adapters;
- metrics, result sinks, and auxiliary knowledge-graph lookup.

The public package focuses on evaluation APIs, fixed benchmark data, and custom attack execution.

## Install

```bash
python -m pip install gpe-eval==0.2.1
```

For repository examples:

```bash
git clone https://github.com/Aquarids/GPE-pub.git
cd GPE-pub
conda env create -f environment.yml
conda activate gpe
```

Copy `.env.example` to `.env` and configure the API endpoint used by the verification method or dynamic attack generator.

Validate a source checkout with `python exp/validate_release.py`.

## Dataset

The package bundles `gpe/data/gpe.jsonl` and `gpe/data/gpe.schema.json`.

- 638 claims in six categories and six reference-label classes
- 5,833 benign evidence objects
- 6,140 related distractors
- 17,120 fixed poisoned evidence objects across four attacks

Each evidence object carries its own `source_scope`, and source filtering reads this attribute directly.

```python
from gpe.resources import bundled_data_path, bundled_schema_path

print(bundled_data_path())
print(bundled_schema_path())
```

## Claim and controlled-evidence access

```python
from gpe import ClaimLoader, DatasetEvidenceLoader

claims = ClaimLoader()
claim = claims.get_claim("benchmark-000002")

evidence = DatasetEvidenceLoader()
clean = evidence.get_evidence_list(
    claim["claim_id"],
    top_k=3,
    poison_ratio=0,
)
attacked = evidence.get_evidence_list(
    claim["claim_id"],
    top_k=3,
    poison_ratio=2 / 3,
    attack_type="ata",
    seed=0,
)
```

`fakegpt`, `poisonedrag`, `ata`, and `ignore` are valid attack names. Ratios replace `round(top_k * ratio)` evidence objects.

## Offline retrieval

```python
from gpe import EvidenceRetriever

retriever = EvidenceRetriever(
    evidence_types={"benign", "related_distractor"},
)

global_hits = retriever.search(
    "measles vaccination coverage",
    top_k=5,
    source_scope="academic",
    mode="bm25",
)

claim_hits = retriever.search(
    "measles vaccination coverage",
    top_k=5,
    claim_id="benchmark-000401",
    source_scope="web",
    mode="bm25",
)
```

Valid source scopes are `web`, `politifact`, `wiki`, `academic`, `social`, and `other`. Source-specific searches can return an empty list, reflecting the coverage in the frozen public-Web snapshot.

Source distributions use evidence attributes directly:

```python
from gpe import EvidenceRetriever, list_source_scopes

benign = EvidenceRetriever(evidence_types={"benign"})
source_counts = {
    scope: benign.count(source_scope=scope)
    for scope in list_source_scopes()
}
print(source_counts)
```

Use `evidence_type` to separate benign, related-distractor, and poisoned records. `retrieval_source` optionally separates individual platforms such as X, Reddit, and Facebook inside the normalized `social` scope. `contents` may be empty when a source exposes only a title or search snippet.

LLM reranking uses opaque candidate numbers and sends only titles, summaries, and keywords to the ranking model.

## Method-visible evidence

The benchmark JSONL retains attack annotations for controlled pool construction and scoring. Before a verification method runs, GPE creates a public evidence view containing document content and observable metadata such as URL, publisher, source scope, author, publication time, account type, revision ID, or submission ID.

```python
from gpe import public_evidence_view

visible = public_evidence_view(raw_evidence, source_scope="academic")
```

The method view excludes benchmark IDs, evidence roles, attack types, attack targets, ATA source links, synthetic-control markers, and the full publication-variant map.

Poison publication metadata is fixed before evaluation. Web and fact-check conditions use attacker-controlled alias domains; Wikipedia, academic, and social conditions use fixed simulated revision, submission, account, and post identifiers.

## Run evaluations

The executable experiment scripts are part of the source repository. Clone the repository before running these commands; PyPI installations expose the corresponding Python APIs.

Claim-local fixed-evidence smoke test:

```bash
python exp/run_claim_retrieval_evaluation.py \
  --method direct_evidence \
  --smoke-test \
  --threads 1
```

Global-pool source-aware evaluation:

```bash
python exp/run_global_retrieval_evaluation.py \
  --method direct_evidence \
  --ratio 2/3 \
  --attacks fakegpt,poisonedrag,ata,ignore \
  --retrieval-source social \
  --retrieval-mode bm25 \
  --top-k 3 \
  --pool-top-k 3 \
  --include-distractors
```

Add `--generate-missing-poison` when evaluating a custom dataset whose fixed attack pool is incomplete. Generated records are cached under `output/dynamic_poison/` and reused on later runs.

## Live retrieval and graph lookup

```python
from gpe import GPE, GPEKnowledgeGraph

benchmark = GPE()
print(benchmark.list_search_sources())
documents = benchmark.search("H5N1 dairy cattle", top_k=5, source="arxiv")

graph = GPEKnowledgeGraph()
print(graph.find_entities("BBC", limit=5))
print(graph.claim_context("benchmark-000001"))
```

The graph is an auxiliary lookup resource. Verification methods may use the JSONL evidence directly without using the graph.

## Citation

```bibtex
@article{wang2026gpe,
  title={GPE: Evaluating GEO Poisoning in Controlled Evidence Environments},
  author={Wang, Zhaoqi and Zhang, Zijian and Yuan, Xiaomei and Kou, Pengtao and Liu, Jiamou and Li, Zhen and Zhu, Liehuang},
  journal={arXiv preprint arXiv:2607.20730},
  year={2026}
}
```

## License

The framework is released under the [MIT License](LICENSE).
