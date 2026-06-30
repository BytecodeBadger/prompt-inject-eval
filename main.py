def main() -> None:
    print("Prompt injection guardrail project initialized.")
    print("Run pipeline steps:")
    print("  1) PYTHONPATH=src uv run python scripts/prepare_data.py")
    print("  2) uv run PYTHONPATH=src python scripts/train_token_classifier.py")
    print("  3) uv run PYTHONPATH=src python scripts/precompute_embeddings.py")
    print("  4) uv run PYTHONPATH=src python scripts/run_evaluation.py")


if __name__ == "__main__":
    main()
