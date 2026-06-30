from guardrail.device import resolve_device


def test_explicit_cpu_selection() -> None:
    result = resolve_device("cpu")
    assert result.selected == "cpu"
