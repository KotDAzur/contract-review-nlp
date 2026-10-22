import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contract_review.classification import task as classification_task
from contract_review.data import task as data_task
from contract_review.evaluation import task
from contract_review.utils import Context

TEST_DATA = Path(__file__).parent / "test_data"


@pytest.fixture
def trained_ctx(tmp_path, tiny_bert, monkeypatch, load_fixture_ledgar):
    """Run déjà passé par les étapes data et classification (réelles, sur le mini-LEDGAR)."""
    ledgar = load_fixture_ledgar(TEST_DATA / "ledgar")
    for module in (data_task, classification_task, task):
        monkeypatch.setattr(module, "load_ledgar", lambda cfg: ledgar)
    cfg = {"seed": 42, "data": {"dataset": "fake", "subset": "x", "revision": "r1"}}
    cfg["classification"] = {
        "model": str(tiny_bert), "max_len": 16, "batch_size": 2, "eval_batch_size": 4,
        "epochs": 1, "lr": 1e-3, "weight_decay": 0.01, "warmup_ratio": 0.1, "grad_clip": 1.0,
        "patience": 1, "fp16": False, "class_weights": None,
    }  # fmt: skip
    cfg["evaluation"] = {"n_bootstrap": 20, "target_precision": 0.9, "ece_bins": 5}
    cfg["synonym_groups"] = [["a", "b"]]
    ctx = Context(cfg=cfg, run_dir=tmp_path, log=logging.getLogger("test"))
    data_task.run(ctx)
    classification_task.run(ctx)
    return ctx


def test_predict(trained_ctx, load_fixture_ledgar):
    ledgar = load_fixture_ledgar(TEST_DATA / "ledgar")

    predictions = task.predict(trained_ctx, ledgar)

    assert set(predictions) == {"validation", "test"}
    assert list(predictions["test"].columns) == ["label", "pred_baseline", "pred", "confidence"]
    assert len(predictions["test"]) == 6


def test_score_models():
    data = pd.read_csv(TEST_DATA / "predictions_full.csv")

    results = task.score_models(data, data, np.arange(3), n_boot=50, seed=0)

    assert results["baseline"]["test"]["accuracy"]["value"] == 1.0
    assert results["transformer"]["test"]["accuracy"]["value"] == 0.7
    assert results["transformer_minus_baseline"]["accuracy"]["value"] == pytest.approx(-0.3)


def test_confidence_and_correct():
    data = pd.read_csv(TEST_DATA / "predictions_full.csv")

    confidence, correct = task.confidence_and_correct(data)

    assert confidence[:2].tolist() == [0.99, 0.95]
    assert correct.sum() == 7


def test_run(trained_ctx):
    task.run(trained_ctx)

    metrics = trained_ctx.run_dir / "metrics"
    assert {"transformer", "baseline", "transformer_minus_baseline"} <= set(
        pd.read_json(metrics / "evaluation.json").columns
    )
    assert len(pd.read_csv(metrics / "per_class.csv")) == 3
    assert (metrics / "predictions_test.parquet").exists()
    assert (trained_ctx.run_dir / "figures" / "reliability.png").exists()
