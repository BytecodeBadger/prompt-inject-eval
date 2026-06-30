from evaluation.metrics import compute_binary_metrics


def test_metric_shapes() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_score = [0.1, 0.6, 0.8, 0.9]

    out = compute_binary_metrics(y_true, y_pred, y_score)
    assert "confusion_matrix" in out
    assert set(out["confusion_matrix"].keys()) == {"tn", "fp", "fn", "tp"}
