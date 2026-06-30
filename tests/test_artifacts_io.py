from pathlib import Path

import numpy as np

from artifacts.io import load_json, save_embeddings, save_json


def test_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b.json"
    payload = {"x": 1}
    save_json(path, payload)
    assert load_json(path) == payload


def test_embedding_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npz"
    matrix = np.array([[1.0, 2.0]], dtype=np.float32)
    save_embeddings(path, matrix)
    out = np.load(path, allow_pickle=False)["embeddings"]
    assert out.shape == (1, 2)
