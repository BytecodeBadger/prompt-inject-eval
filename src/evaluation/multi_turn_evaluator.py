from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TurnResult:
    session_id: str
    turn_index: int
    decision: str
    expected_adversarial: int


def evaluate_sessions(pipeline: Any, sessions: list[dict[str, Any]]) -> dict[str, Any]:
    turn_results: list[TurnResult] = []
    session_blocked = 0

    for session in sessions:
        sid = session["session_id"]
        blocked = False
        for turn in session["turns"]:
            outcome = pipeline.evaluate_prompt(turn["text"])
            decision = outcome["decision"]
            if decision == "block" and turn["label"] == 1:
                blocked = True
            turn_results.append(
                TurnResult(
                    session_id=sid,
                    turn_index=int(turn["turn_index"]),
                    decision=decision,
                    expected_adversarial=int(turn["label"]),
                )
            )
        if blocked:
            session_blocked += 1

    return {
        "turn_level": [r.__dict__ for r in turn_results],
        "session_level": {
            "total_sessions": len(sessions),
            "blocked_sessions": session_blocked,
            "session_block_rate": (session_blocked / len(sessions)) if sessions else 0.0,
        },
    }
