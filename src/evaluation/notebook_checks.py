from __future__ import annotations

from pathlib import Path


REQUIRED_ARTIFACTS = [
    "artifacts/metrics/llm_before_after_summary.json",
    "artifacts/metrics/llm_before_after_runtime.json",
    "artifacts/predictions/llm_before_after.jsonl",
]


def assert_notebook_prerequisites(root: Path) -> None:
    missing = [p for p in REQUIRED_ARTIFACTS if not (root / p).exists()]
    if missing:
        msg = "Missing required artifacts for notebook execution: " + ", ".join(missing)
        raise FileNotFoundError(msg)
