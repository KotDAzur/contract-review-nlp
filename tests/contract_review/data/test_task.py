import json
import logging
from pathlib import Path

import pandas as pd

from contract_review.data import task
from contract_review.utils import Context

TEST_DATA = Path(__file__).parent / "test_data"
DATA_CFG = {"dataset": "fake/ledgar", "subset": "x", "revision": "r1"}


def test_load_ledgar(monkeypatch):
    calls = []  # le téléchargement Hugging Face est la seule frontière remplacée
    monkeypatch.setattr(task, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or "dataset")

    assert task.load_ledgar(DATA_CFG) == "dataset"
    assert calls == [(("fake/ledgar", "x"), {"revision": "r1"})]


def test_run(tmp_path, monkeypatch, load_fixture_ledgar):
    monkeypatch.setattr(task, "load_ledgar", lambda cfg: load_fixture_ledgar(TEST_DATA / "ledgar"))
    ctx = Context(cfg={"data": DATA_CFG}, run_dir=tmp_path, log=logging.getLogger("test"))

    task.run(ctx)

    data_dir = tmp_path / "data"
    stats = json.loads((data_dir / "stats.json").read_text())
    assert stats["sizes"] == {"train": 6, "validation": 6, "test": 6}
    assert json.loads((data_dir / "label_names.json").read_text()) == ["a", "b", "c"]
    assert json.loads((data_dir / "label_counts_train.json").read_text()) == {
        "a": 2,
        "b": 2,
        "c": 2,
    }
    assert len(pd.read_parquet(data_dir / "slices_test.parquet")) == 6
