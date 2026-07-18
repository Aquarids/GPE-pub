from gpe.dataloader import ClaimLoader, DatasetEvidenceLoader
from gpe.exp import EvaluationPipeline, run_prediction
from gpe.labels import Label, parse_label
from gpe.methods import DETECTORS
from gpe.metrics import Evaluator, label_score, token_efficiency
from gpe.reporting import compute_metrics_from_jsonl
from gpe.search import Searcher
from gpe.poison import PoisonCache
from gpe.resources import GPE_DATA_PATH


class GPE:

    def __init__(
        self,
        data_path=None,
        poison_path=None,
        llm=None,
        logger=None,
        dynamic_poison_path=None,
    ):
        self.data_path = data_path or GPE_DATA_PATH
        self.poison_path = poison_path
        self.dynamic_poison_path = dynamic_poison_path
        self.llm = llm
        self.logger = logger
        self.claims = ClaimLoader(self.data_path)
        self.evidence = DatasetEvidenceLoader(self.data_path, poison_path)
        self.poison_cache = PoisonCache(
            poison_path or self.data_path,
            dynamic_path=dynamic_poison_path,
            llm=llm,
            logger=logger,
        )
        self.searcher = Searcher(self.claims, logger=logger)
        self.evaluator = Evaluator(self.claims)

    def list_methods(self):
        return sorted(DETECTORS)

    def list_search_sources(self):
        return self.searcher.list_sources()

    def create_method(self, name, config=None):
        if self.llm is None:
            raise ValueError("creating a built-in method requires an LLMWrapper instance")
        if name not in DETECTORS:
            raise KeyError(f"unknown method: {name}; available={sorted(DETECTORS)}")
        return DETECTORS[name](self.logger, self.llm, config or {})

    def predict(self, method, claim, evidence=None, method_config=None):
        """Run one built-in method on a claim and return prediction plus token usage."""
        detector = self.create_method(method, method_config)
        prediction, usage = run_prediction(detector, self.llm, claim, evidence or [])
        return {"method": method, "prediction": prediction, "usage": usage}

    def predict_claim(
        self,
        method,
        claim_id,
        evidence_source="dataset",
        poison_ratio=0.0,
        attack_type=None,
        top_k=None,
        seed=0,
        generate_missing_poison=True,
        method_config=None,
    ):
        """Run one built-in method on a benchmark claim with dataset or external-search evidence."""
        claim = self.claims.get_claim(claim_id)
        evidence = []
        if evidence_source == "dataset":
            evidence = self.evidence.get_evidence_list(
                claim_id,
                top_k,
                poison_ratio,
                attack_type,
                seed,
                poison_cache=self.poison_cache,
                generate_missing_poison=generate_missing_poison,
            )
        config = dict(method_config or {})
        config["evidence_source"] = evidence_source
        result = self.predict(method, claim["original_claim"], evidence=evidence, method_config=config)
        result.update({"claim_id": claim_id, "evidence_source": evidence_source, "poison_ratio": poison_ratio, "attack_type": attack_type})
        return result

    def search(self, query, top_k=5, source="web", extra=None):
        return self.searcher.search(query, top_k=top_k, source=source, extra=extra)

    def create_pipeline(self):
        if self.llm is None:
            raise ValueError("creating an evaluation pipeline requires an LLMWrapper instance")
        return EvaluationPipeline(
            self.claims,
            self.evidence,
            self.evaluator,
            self.llm,
            self.poison_cache,
        )

    def compute_metrics(self, input_path, output_path=None):
        return compute_metrics_from_jsonl(input_path, output_path)


__all__ = ["GPE", "EvaluationPipeline", "ClaimLoader", "DatasetEvidenceLoader", "Searcher", "Evaluator", "Label", "parse_label", "label_score", "token_efficiency"]
