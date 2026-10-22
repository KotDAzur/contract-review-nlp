from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from contract_review.data.process import group_ids
from contract_review.evaluation import process

TEST_DATA = Path(__file__).parent / "test_data"
NAMES = ["a", "b", "c"]


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TEST_DATA / name)


@pytest.fixture
def predictions():
    return read_csv("predictions.csv")


def test_metrics(predictions):
    result = process.metrics(predictions["label"], predictions["pred"])

    assert result["accuracy"] == pytest.approx(0.7)
    assert result["micro_f1"] == result["accuracy"]
    assert result["macro_f1"] == pytest.approx(0.69841, abs=1e-4)
    assert result["weighted_f1"] == pytest.approx(0.68571, abs=1e-4)


def test_bootstrap_metrics(predictions):
    first = process.bootstrap_metrics(predictions["label"], predictions["pred"], 50, seed=0)
    second = process.bootstrap_metrics(predictions["label"], predictions["pred"], 50, seed=0)

    assert first.shape == (50, 4)
    assert list(first.columns) == ["accuracy", "micro_f1", "macro_f1", "weighted_f1"]
    pd.testing.assert_frame_equal(first, second)


def test_with_ci():
    result = process.with_ci({"accuracy": 0.7}, read_csv("bootstrap_samples.csv"))

    assert result == {"accuracy": {"value": 0.7, "ci": [0.025, 0.975]}}


def test_tolerant_accuracy(predictions):
    ids = group_ids(NAMES, [["a", "c"]])

    result = process.tolerant_accuracy(
        predictions["label"].to_numpy(), predictions["pred"].to_numpy(), ids
    )

    assert result == pytest.approx(0.9)


def test_per_class(predictions):
    result = process.per_class(predictions["label"], predictions["pred"], NAMES)

    pd.testing.assert_frame_equal(
        result, read_csv("expected_per_class.csv"), atol=1e-5, check_dtype=False
    )


def test_top_confusions(predictions):
    result = process.top_confusions(
        predictions["label"].to_numpy(), predictions["pred"].to_numpy(), NAMES
    )

    result = result.sort_values(["true", "pred"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, read_csv("expected_confusions.csv"), check_dtype=False)


def test_reliability(predictions):
    confidence = predictions["confidence"].to_numpy()
    correct = (predictions["label"] == predictions["pred"]).to_numpy()

    table, ece = process.reliability(confidence, correct, n_bins=2)

    pd.testing.assert_frame_equal(table, read_csv("expected_reliability.csv"), check_dtype=False)
    assert ece == pytest.approx(0.064)


def test_coverage_precision(predictions):
    confidence = predictions["confidence"].to_numpy()
    correct = (predictions["label"] == predictions["pred"]).to_numpy()

    curve = process.coverage_precision(confidence, correct)

    pd.testing.assert_frame_equal(
        curve, read_csv("expected_curve.csv"), atol=1e-5, check_dtype=False
    )


def test_choose_threshold():
    assert process.choose_threshold(read_csv("expected_curve.csv"), target=0.9) == 0.8


def test_at_threshold(predictions):
    confidence = predictions["confidence"].to_numpy()
    correct = (predictions["label"] == predictions["pred"]).to_numpy()

    assert process.at_threshold(confidence, correct, 0.8) == {"coverage": 0.5, "precision": 1.0}


def test_slice_accuracy(predictions):
    correct = (predictions["label"] == predictions["pred"]).to_numpy()

    table = process.slice_accuracy(correct, read_csv("slices.csv"), n_boot=100, seed=0)

    assert table["value"].tolist() == ["a", "b"]
    assert table["n"].tolist() == [5, 5]
    assert table["accuracy"].tolist() == pytest.approx([1.0, 0.4])
    assert (table["ci_low"] <= table["accuracy"]).all()
    assert (table["accuracy"] <= table["ci_high"]).all()


def test_plot_weakest():
    fig = process.plot_weakest(read_csv("expected_per_class.csv"))

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Les 15 types de clause les plus faibles"


def test_plot_reliability():
    fig = process.plot_reliability(read_csv("expected_reliability.csv"), ece=0.064)

    assert len(fig.axes[0].lines) == 2  # diagonale parfaite + courbe du modèle


def test_plot_coverage():
    point = {"coverage": 0.5, "precision": 1.0}

    fig = process.plot_coverage(read_csv("expected_curve.csv"), 0.9, point)

    assert len(fig.axes[0].lines) == 3  # courbe, précision visée, seuil choisi
