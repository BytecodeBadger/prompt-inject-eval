from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DeviceSelection:
    requested: str | None
    selected: str
    reason: str


def _normalize_device_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = name.strip().lower()
    return normalized or None


def resolve_device(explicit: str | None = None) -> DeviceSelection:
    """Resolve execution device with precedence: explicit > env > auto."""
    requested = _normalize_device_name(explicit) or _normalize_device_name(os.getenv("GUARDRAIL_DEVICE"))

    try:
        import torch
    except ImportError:
        if requested and requested not in {"cpu"}:
            return DeviceSelection(requested=requested, selected="cpu", reason="torch_unavailable_fallback_to_cpu")
        return DeviceSelection(requested=requested, selected="cpu", reason="torch_unavailable")

    available = {
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }

    if requested:
        if requested in available and available[requested]:
            return DeviceSelection(requested=requested, selected=requested, reason="requested_available")
        auto_pick = "cuda" if available["cuda"] else "mps" if available["mps"] else "cpu"
        return DeviceSelection(requested=requested, selected=auto_pick, reason=f"requested_unavailable_fallback_to_{auto_pick}")

    auto_pick = "cuda" if available["cuda"] else "mps" if available["mps"] else "cpu"
    return DeviceSelection(requested=None, selected=auto_pick, reason="auto_detect")
