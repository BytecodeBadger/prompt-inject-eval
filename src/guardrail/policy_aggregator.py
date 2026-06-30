from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyDecision:
    allow: bool
    reason: str
    similarity_score: float
    classifier_probability: float
    confidence: float


@dataclass
class PolicyConfig:
    similarity_block_threshold: float = 1.1
    classifier_block_threshold: float = 0.56
    weighted_block_threshold: float = 1.1
    similarity_weight: float = 0.5
    classifier_weight: float = 0.5


class PolicyAggregator:
    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def decide(self, similarity_score: float, classifier_probability: float) -> PolicyDecision:
        weighted = (
            self.config.similarity_weight * similarity_score
            + self.config.classifier_weight * classifier_probability
        )

        if similarity_score >= self.config.similarity_block_threshold:
            return PolicyDecision(
                allow=False,
                reason="blocked_similarity_threshold",
                similarity_score=similarity_score,
                classifier_probability=classifier_probability,
                confidence=similarity_score,
            )

        if classifier_probability >= self.config.classifier_block_threshold:
            return PolicyDecision(
                allow=False,
                reason="blocked_classifier_threshold",
                similarity_score=similarity_score,
                classifier_probability=classifier_probability,
                confidence=classifier_probability,
            )

        if weighted >= self.config.weighted_block_threshold:
            return PolicyDecision(
                allow=False,
                reason="blocked_weighted_threshold",
                similarity_score=similarity_score,
                classifier_probability=classifier_probability,
                confidence=weighted,
            )

        return PolicyDecision(
            allow=True,
            reason="allowed",
            similarity_score=similarity_score,
            classifier_probability=classifier_probability,
            confidence=1.0 - weighted,
        )
