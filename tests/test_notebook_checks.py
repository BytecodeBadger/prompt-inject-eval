from pathlib import Path

import pytest

from evaluation.notebook_checks import assert_notebook_prerequisites


def test_notebook_check_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        assert_notebook_prerequisites(tmp_path)
