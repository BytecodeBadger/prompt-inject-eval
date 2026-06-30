from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from artifacts.io import load_embeddings, load_jsonl, save_json, save_jsonl
from evaluation.llm_attack_eval import assess_attack_success, summarize_before_after
from guardrail.device import resolve_device
from guardrail.pipeline import GuardrailPipeline
from guardrail.telemetry import JsonlLogger, finish_runtime, print_runtime_summary, start_runtime

ROOT = Path(__file__).resolve().parents[1]


def _map_device_for_torch(device: str) -> str:
    if device == "cuda":
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


def _load_llm(model_id: str, device: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(_map_device_for_torch(device))
    model.eval()
    return tokenizer, model


def _generate_response(tokenizer: Any, model: Any, prompt: str, device: str, max_new_tokens: int) -> str:
    wrapped = (
        "You are an assistant. Answer directly.\n"
        f"User: {prompt}\n"
        "Assistant:"
    )
    inputs = tokenizer(wrapped, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(_map_device_for_torch(device)) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real LLM before/after guardrail attack evaluation.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    runtime = start_runtime("run_llm_before_after_eval")
    logger = JsonlLogger(ROOT / "logs" / "evaluation.log")
    device = resolve_device(args.device)

    print(f"selected_device={device.selected} reason={device.reason}")
    print(f"llm_model={args.model_id}")

    tokenizer, model = _load_llm(model_id=args.model_id, device=device.selected)

    test_rows = load_jsonl(ROOT / "data" / "processed" / "test.jsonl")
    if args.limit > 0:
        test_rows = test_rows[: args.limit]

    pipeline = GuardrailPipeline.from_artifacts(
        model_path=ROOT / "artifacts" / "models" / "token_classifier.joblib",
        benign_embeddings=load_embeddings(ROOT / "artifacts" / "embeddings" / "benign_embeddings.npz"),
        adversarial_embeddings=load_embeddings(ROOT / "artifacts" / "embeddings" / "adversarial_embeddings.npz"),
        embedding_model_name="hashing://default",
        device=device.selected,
    )

    rows: list[dict[str, Any]] = []
    for row in test_rows:
        prompt = str(row["text"])
        label = int(row["label"])

        t0 = time.perf_counter()
        before_response = _generate_response(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            device=device.selected,
            max_new_tokens=args.max_new_tokens,
        )
        before_latency_ms = (time.perf_counter() - t0) * 1000.0
        before_assessment = assess_attack_success(prompt=prompt, response=before_response, label=label)

        t1 = time.perf_counter()
        guardrail = pipeline.evaluate_prompt(prompt)
        guardrail_latency_ms = (time.perf_counter() - t1) * 1000.0

        after_response: str
        after_latency_ms = 0.0
        if guardrail["decision"] == "block":
            after_response = "[BLOCKED_BY_GUARDRAIL]"
            after_assessment = assess_attack_success(prompt=prompt, response=after_response, label=label)
        else:
            t2 = time.perf_counter()
            after_response = _generate_response(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                device=device.selected,
                max_new_tokens=args.max_new_tokens,
            )
            after_latency_ms = (time.perf_counter() - t2) * 1000.0
            after_assessment = assess_attack_success(prompt=prompt, response=after_response, label=label)

        rows.append(
            {
                **row,
                "before_response": before_response,
                "before_refused": before_assessment.refused,
                "before_compromised": before_assessment.compromised,
                "before_injection_success": before_assessment.success,
                "before_latency_ms": before_latency_ms,
                "guardrail_decision": guardrail["decision"],
                "guardrail_reason": guardrail["decision_reason"],
                "guardrail_confidence": guardrail["confidence"],
                "guardrail_latency_ms": guardrail_latency_ms,
                "after_response": after_response,
                "after_refused": after_assessment.refused,
                "after_compromised": after_assessment.compromised,
                "after_injection_success": after_assessment.success,
                "after_latency_ms": after_latency_ms,
            }
        )

    summary = summarize_before_after(rows)
    summary["run_metadata"] = {
        "model_id": args.model_id,
        "selected_device": device.selected,
        "fallback_reason": device.reason,
    }

    save_jsonl(ROOT / "artifacts" / "predictions" / "llm_before_after.jsonl", rows)
    save_json(ROOT / "artifacts" / "metrics" / "llm_before_after_summary.json", summary)

    runtime_summary = finish_runtime(runtime)
    save_json(ROOT / "artifacts" / "metrics" / "llm_before_after_runtime.json", runtime_summary)

    logger.log(
        {
            "run_id": runtime.run_id,
            "stage": runtime.stage,
            "device": device.selected,
            "elapsed_ms": int(runtime_summary["final_runtime_seconds"] * 1000),
            "sample_count": len(rows),
            "key_metrics_snapshot": {
                "before_success_rate": summary["before"]["success_rate"],
                "after_success_rate": summary["after"]["success_rate"],
                "success_rate_reduction": summary["delta"]["success_rate_reduction"],
            },
            "final_runtime_seconds": runtime_summary["final_runtime_seconds"],
            "message": "llm_before_after_eval_complete",
        }
    )

    print_runtime_summary(runtime_summary)
    print(
        "before_after_metrics: "
        f"before_success_rate={summary['before']['success_rate']:.3f} "
        f"after_success_rate={summary['after']['success_rate']:.3f} "
        f"reduction={summary['delta']['success_rate_reduction']:.3f}"
    )


if __name__ == "__main__":
    main()
