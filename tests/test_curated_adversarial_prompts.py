from evaluation.curated_adversarial_prompts import load_curated_adversarial_prompts


def test_curated_prompt_pack_has_requested_families() -> None:
    prompts = load_curated_adversarial_prompts()
    categories = {row["category"] for row in prompts}

    assert len(prompts) >= 10
    assert "Instruction smuggling" in categories
    assert "Tool-call coercion" in categories
    assert "Multi-turn escalation" in categories
