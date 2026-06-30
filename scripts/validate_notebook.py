from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.notebook_checks import assert_notebook_prerequisites
from guardrail.device import resolve_device
from guardrail.telemetry import JsonlLogger, finish_runtime, print_runtime_summary, start_runtime

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate notebook prerequisites.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    runtime = start_runtime("validate_notebook")
    logger = JsonlLogger(ROOT / "logs" / "evaluation.log")
    device = resolve_device(args.device)

    assert_notebook_prerequisites(ROOT)

    summary = finish_runtime(runtime)
    logger.log(
        {
            "run_id": runtime.run_id,
            "stage": runtime.stage,
            "device": device.selected,
            "elapsed_ms": int(summary["final_runtime_seconds"] * 1000),
            "sample_count": 0,
            "final_runtime_seconds": summary["final_runtime_seconds"],
            "message": "notebook_prereq_ok",
        }
    )

    print(f"selected_device={device.selected} reason={device.reason}")
    print_runtime_summary(summary)
    print("Notebook prerequisite validation passed.")


if __name__ == "__main__":
    main()
