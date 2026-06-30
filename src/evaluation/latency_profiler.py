from __future__ import annotations

import time
from typing import Callable, Any

import numpy as np


class LatencyProfiler:
    def __init__(self):
        self.samples_ms: list[float] = []

    def measure(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.samples_ms.append(elapsed)
        return result

    def summary(self) -> dict[str, float]:
        if not self.samples_ms:
            return {
                "count": 0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "mean_ms": 0.0,
                "throughput_prompts_per_sec": 0.0,
            }

        arr = np.asarray(self.samples_ms)
        mean_ms = float(np.mean(arr))
        return {
            "count": int(arr.size),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": mean_ms,
            "throughput_prompts_per_sec": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        }
