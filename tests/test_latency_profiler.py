from evaluation.latency_profiler import LatencyProfiler


def test_latency_profiler_summary() -> None:
    profiler = LatencyProfiler()

    def noop() -> int:
        return 1

    for _ in range(5):
        profiler.measure(noop)

    summary = profiler.summary()
    assert summary["count"] == 5
    assert summary["p95_ms"] >= 0
