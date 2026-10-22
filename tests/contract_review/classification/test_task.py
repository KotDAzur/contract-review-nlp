import json
import logging
import shutil
from pathlib import Path

import pytest

from contract_review.classification import task
from contract_review.utils import Context

TEST_DATA = Path(__file__).parent / "test_data"


@pytest.fixture
def ctx(tmp_path, tiny_bert):
    cfg = {"seed": 42, "data": {"dataset": "fake", "subset": "x", "revision": "r1"}}
    cfg["classification"] = {
        "model": str(tiny_bert), "max_len": 16, "batch_size": 2, "eval_batch_size": 4,
        "epochs": 1, "lr": 1e-3, "weight_decay": 0.01, "warmup_ratio": 0.1, "grad_clip": 1.0,
        "patience": 1, "fp16": False, "class_weights": None,
    }  # fmt: skip
    return Context(cfg=cfg, run_dir=tmp_path, log=logging.getLogger("test"))


def test_predict_probs(ctx, tiny_bert):
    shutil.copytree(tiny_bert, ctx.run_dir / "models" / "classifier")

    probs = task.predict_probs(ctx, ["the agreement", "notice days"])

    assert probs.shape == (2, 3)
    assert probs.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_run(ctx, monkeypatch, load_fixture_ledgar):
    monkeypatch.setattr(task, "load_ledgar", lambda cfg: load_fixture_ledgar(TEST_DATA / "ledgar"))

    task.run(ctx)

    models = ctx.run_dir / "models"
    assert (models / "baseline.joblib").exists()
    assert (models / "classifier" / "config.json").exists()
    history = json.loads((ctx.run_dir / "metrics" / "history.json").read_text())
    assert [epoch["epoch"] for epoch in history] == [1]
