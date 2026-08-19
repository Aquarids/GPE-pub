# GPE

This repository is the official code implementation of:

> **GPE: Evaluating Robust Evidence Aggregation for Fact Verification under
> Controllable GEO-Style Poisoning**

> **Experimental project.** The code and benchmark are provided for research
> use and may contain bugs. Please consult the paper for the full methodology,
> experimental settings, and interpretation of results.

## Install

```bash
python -m pip install gpe-eval
```

For development, clone the repository and create the included Conda environment:

```bash
conda env create -f environment.yml
conda activate gpe
```

## Use

The graph and retrieval example uses only the bundled data:

```bash
python exp/run_graph_and_retrieval.py \
  --entity BBC \
  --claim-id benchmark-000001 \
  --query "steel production emissions" \
  --top-k 3
```

To run the claim-local or global retrieval evaluation smoke test, configure
`LLM_API_KEY` in `.env`, then run one of:

```bash
python exp/run_claim_retrieval_evaluation.py \
  --method direct_evidence \
  --smoke-test \
  --threads 1

python exp/run_global_retrieval_evaluation.py \
  --method direct_evidence \
  --smoke-test \
  --threads 1
```

## Citation

If you find this repository or the GPE benchmark useful, please cite:

```bibtex
@article{wang2026gpe,
  title={GPE: Evaluating Robust Evidence Aggregation for Fact Verification under Controllable GEO-Style Poisoning},
  author={Wang, Zhaoqi and Zhang, Zijian and Yuan, Xiaomei and Kou, Pengtao and Liu, Jiamou and Li, Zhen and Zhu, Liehuang},
  journal={arXiv preprint arXiv:2607.20730},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
