from __future__ import annotations

import argparse
from pathlib import Path

from artifacts.io import load_jsonl, save_embeddings, save_json
from guardrail.device import resolve_device
from guardrail.embedding_layer import EmbeddingLayer
from guardrail.telemetry import JsonlLogger, finish_runtime, print_runtime_summary, start_runtime

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute reference embeddings.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--force-hashing", action="store_true")
    args = parser.parse_args()

    runtime = start_runtime("precompute_embeddings")
    logger = JsonlLogger(ROOT / "logs" / "pipeline.log")
    device = resolve_device(args.device)

    train_rows = load_jsonl(ROOT / "data" / "processed" / "train.jsonl")
    benign_texts = [r["text"] for r in train_rows if int(r["label"]) == 0]
    adversarial_texts = [r["text"] for r in train_rows if int(r["label"]) == 1]

    model_name = "hashing://default" if args.force_hashing else args.model_name
    layer = EmbeddingLayer(model_name=model_name, device=device.selected)
    benign_embeddings = layer.encode(benign_texts)
    adversarial_embeddings = layer.encode(adversarial_texts)

    save_embeddings(ROOT / "artifacts" / "embeddings" / "benign_embeddings.npz", benign_embeddings)
    save_embeddings(ROOT / "artifacts" / "embeddings" / "adversarial_embeddings.npz", adversarial_embeddings)
    save_json(
        ROOT / "artifacts" / "embeddings" / "index_metadata.json",
        {
            "embedding_model": layer.model_info.model_name,
            "backend": layer.model_info.backend,
            "version": layer.model_info.version,
            "selected_device": device.selected,
            "fallback_reason": device.reason,
            "benign_count": len(benign_texts),
            "adversarial_count": len(adversarial_texts),
        },
    )

    summary = finish_runtime(runtime)
    print(f"selected_device={device.selected} reason={device.reason} backend={layer.model_info.backend}")
    print_runtime_summary(summary)

    logger.log(
        {
            "run_id": runtime.run_id,
            "stage": runtime.stage,
            "device": device.selected,
            "elapsed_ms": int(summary["final_runtime_seconds"] * 1000),
            "sample_count": len(train_rows),
            "key_metrics_snapshot": {
                "benign_embeddings": len(benign_texts),
                "adversarial_embeddings": len(adversarial_texts),
                "backend": layer.model_info.backend,
            },
            "final_runtime_seconds": summary["final_runtime_seconds"],
            "message": "embedding_precompute_complete",
        }
    )


if __name__ == "__main__":
    main()
