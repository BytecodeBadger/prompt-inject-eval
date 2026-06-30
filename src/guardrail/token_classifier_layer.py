from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class TokenClassifierResult:
    probability_adversarial: float
    predicted_label: int


class TokenClassifierLayer:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.pipeline: Any = joblib.load(self.model_path)

    def predict_one(self, text: str) -> TokenClassifierResult:
        probs = self.pipeline.predict_proba([text])[0]
        pred = int(np.argmax(probs))
        return TokenClassifierResult(probability_adversarial=float(probs[1]), predicted_label=pred)
