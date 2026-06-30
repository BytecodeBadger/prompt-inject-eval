from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
from typing import Any


@dataclass
class RuntimeContext:
    run_id: str
    stage: str
    started_at: float
    started_iso: str


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        row = dict(payload)
        row.setdefault("event_time", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def start_runtime(stage: str) -> RuntimeContext:
    started_at = time.perf_counter()
    return RuntimeContext(
        run_id=str(uuid.uuid4()),
        stage=stage,
        started_at=started_at,
        started_iso=datetime.now(timezone.utc).isoformat(),
    )


def finish_runtime(ctx: RuntimeContext) -> dict[str, Any]:
    finished_at = time.perf_counter()
    elapsed = finished_at - ctx.started_at
    return {
        **asdict(ctx),
        "finished_iso": datetime.now(timezone.utc).isoformat(),
        "final_runtime_seconds": elapsed,
        "final_runtime_human": f"{elapsed:.2f}s",
    }


def print_runtime_summary(summary: dict[str, Any]) -> None:
    print(f"run_id={summary['run_id']} stage={summary['stage']}")
    print(f"start={summary['started_iso']} end={summary['finished_iso']}")
    print(
        "final_runtime="
        f"{summary['final_runtime_seconds']:.2f}s "
        f"({summary['final_runtime_human']})"
    )
