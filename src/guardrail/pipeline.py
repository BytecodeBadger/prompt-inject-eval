from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .embedding_layer import EmbeddingLayer, cosine_similarity_batch
from .policy_aggregator import PolicyAggregator
from .token_classifier_layer import TokenClassifierLayer


class GuardrailPipeline:
    def __init__(
        self,
        embedding_layer: EmbeddingLayer,
        classifier_layer: TokenClassifierLayer,
        policy_aggregator: PolicyAggregator,
        benign_reference_embeddings: np.ndarray,
        adversarial_reference_embeddings: np.ndarray,
    ):
        self.embedding_layer = embedding_layer
        self.classifier_layer = classifier_layer
        self.policy_aggregator = policy_aggregator
        self.benign_reference_embeddings = benign_reference_embeddings
        self.adversarial_reference_embeddings = adversarial_reference_embeddings

    def evaluate_prompt(self, text: str) -> dict[str, Any]:
        query_vec = self.embedding_layer.encode([text])
        adv_scores = cosine_similarity_batch(query_vec, self.adversarial_reference_embeddings)
        benign_scores = cosine_similarity_batch(query_vec, self.benign_reference_embeddings)

        adv_similarity = float(np.max(adv_scores)) if adv_scores.size else 0.0
        benign_similarity = float(np.max(benign_scores)) if benign_scores.size else 0.0
        normalized_similarity = max(0.0, min(1.0, (adv_similarity - benign_similarity + 1.0) / 2.0))

        cls = self.classifier_layer.predict_one(text)
        decision = self.policy_aggregator.decide(
            similarity_score=normalized_similarity,
            classifier_probability=cls.probability_adversarial,
        )

        return {
            "decision": "allow" if decision.allow else "block",
            "decision_reason": decision.reason,
            "similarity_score": normalized_similarity,
            "max_adv_similarity": adv_similarity,
            "max_benign_similarity": benign_similarity,
            "classifier_probability": cls.probability_adversarial,
            "classifier_label": cls.predicted_label,
            "confidence": decision.confidence,
            "metadata": asdict(decision),
        }

    @classmethod
    def from_artifacts(
        cls,
        model_path: str | Path,
        benign_embeddings: np.ndarray,
        adversarial_embeddings: np.ndarray,
        embedding_model_name: str,
        device: str,
    ) -> "GuardrailPipeline":
        embedding_layer = EmbeddingLayer(model_name=embedding_model_name, device=device)
        classifier_layer = TokenClassifierLayer(model_path=model_path)
        policy_aggregator = PolicyAggregator()
        return cls(
            embedding_layer=embedding_layer,
            classifier_layer=classifier_layer,
            policy_aggregator=policy_aggregator,
            benign_reference_embeddings=benign_embeddings,
            adversarial_reference_embeddings=adversarial_embeddings,
        )
