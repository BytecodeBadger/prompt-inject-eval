from __future__ import annotations

import argparse
from pathlib import Path

from artifacts.io import load_embeddings, load_json
from guardrail.device import resolve_device
from guardrail.pipeline import GuardrailPipeline

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-shot prompt guardrail inference.")
    parser.add_argument("prompt", help="Prompt text to evaluate")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    embedding_meta = load_json(ROOT / "artifacts" / "embeddings" / "index_metadata.json")
    pipeline = GuardrailPipeline.from_artifacts(
        model_path=ROOT / "artifacts" / "models" / "token_classifier.joblib",
        benign_embeddings=load_embeddings(ROOT / "artifacts" / "embeddings" / "benign_embeddings.npz"),
        adversarial_embeddings=load_embeddings(ROOT / "artifacts" / "embeddings" / "adversarial_embeddings.npz"),
        embedding_model_name=str(embedding_meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        device=device.selected,
    )
    result = pipeline.evaluate_prompt(args.prompt)
    print(f"selected_device={device.selected} reason={device.reason}")
    print(result)


if __name__ == "__main__":
    main()
