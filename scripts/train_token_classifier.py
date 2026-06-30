from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from artifacts.io import load_jsonl, save_json, save_model
from evaluation.metrics import compute_binary_metrics
from guardrail.device import resolve_device
from guardrail.telemetry import JsonlLogger, finish_runtime, print_runtime_summary, start_runtime

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight token classifier.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    runtime = start_runtime("train_token_classifier")
    logger = JsonlLogger(ROOT / "logs" / "pipeline.log")
    device = resolve_device(args.device)

    train_rows = load_jsonl(ROOT / "data" / "processed" / "train.jsonl")
    val_rows = load_jsonl(ROOT / "data" / "processed" / "validation.jsonl")

    x_train = [r["text"] for r in train_rows]
    y_train = [int(r["label"]) for r in train_rows]
    x_val = [r["text"] for r in val_rows]
    y_val = [int(r["label"]) for r in val_rows]

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=40000)),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")),
        ]
    )
    model.fit(x_train, y_train)

    val_probs = model.predict_proba(x_val)[:, 1].tolist()
    val_pred = [1 if p >= 0.5 else 0 for p in val_probs]
    val_metrics = compute_binary_metrics(y_val, val_pred, val_probs)

    save_model(ROOT / "artifacts" / "models" / "token_classifier.joblib", model)
    save_json(
        ROOT / "artifacts" / "models" / "model_metadata.json",
        {
            "model_name": "tfidf_logreg_lightweight",
            "version": "1",
            "selected_device": device.selected,
            "fallback_reason": device.reason,
            "train_samples": len(train_rows),
            "validation_samples": len(val_rows),
            "validation_metrics": val_metrics,
        },
    )

    summary = finish_runtime(runtime)
    print(f"selected_device={device.selected} reason={device.reason}")
    print_runtime_summary(summary)

    logger.log(
        {
            "run_id": runtime.run_id,
            "stage": runtime.stage,
            "device": device.selected,
            "elapsed_ms": int(summary["final_runtime_seconds"] * 1000),
            "sample_count": len(train_rows),
            "key_metrics_snapshot": {
                "val_f1": val_metrics["f1"],
                "val_recall": val_metrics["recall"],
            },
            "final_runtime_seconds": summary["final_runtime_seconds"],
            "message": "training_complete",
        }
    )


if __name__ == "__main__":
    main()
