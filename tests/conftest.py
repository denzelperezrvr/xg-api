import pytest


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    monkeypatch.setenv("XG_LOG_PATH", str(tmp_path / "predictions.jsonl"))