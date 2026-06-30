from guardrail.policy_aggregator import PolicyAggregator, PolicyConfig


def test_blocks_by_similarity_threshold() -> None:
    agg = PolicyAggregator(PolicyConfig(similarity_block_threshold=0.7, classifier_block_threshold=0.9))
    result = agg.decide(similarity_score=0.8, classifier_probability=0.1)
    assert result.allow is False
    assert result.reason == "blocked_similarity_threshold"


def test_allows_low_scores() -> None:
    agg = PolicyAggregator()
    result = agg.decide(similarity_score=0.1, classifier_probability=0.2)
    assert result.allow is True
