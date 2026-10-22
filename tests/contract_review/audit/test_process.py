from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contract_review.audit import process

TEST_DATA = Path(__file__).parent / "test_data"


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(TEST_DATA / name, **kwargs)


@pytest.fixture
def predictions():
    return read_csv("predictions.csv")


@pytest.fixture
def report_inputs():
    """Tableaux d'entrée du rapport : tranches, effets, hors distribution, fuite."""
    return {
        "gaps": read_csv("gaps.csv"),
        "counterfactuals": read_csv("counterfactuals.csv"),
        "ood": read_csv("ood.csv"),
        "leakage": read_csv("leakage.csv"),
        "cfg": {"alpha": 0.05},
    }


def test_match_case():
    assert process.match_case("she", "He") == "She"


def test_swap_gender():
    text = "The Employee shall give his notice and the Company shall pay him."

    assert process.swap_gender(text, "feminine") == (
        "The Employee shall give her notice and the Company shall pay her."
    )


def test_replace_jurisdiction():
    text = "governed by the laws of the State of New York."

    assert process.replace_jurisdiction(text, "France") == "governed by the laws of France."


def test_replace_company_form():
    result = process.replace_company_form("Acme, Inc. and Beta LLC agree.", "GmbH")

    assert result == "Acme, GmbH and Beta GmbH agree."


def test_perturb():
    texts = ["a Delaware b", "rien à changer"]

    changed = process.perturb(texts, lambda t: process.replace_jurisdiction(t, "Texas"))

    assert changed == ([0], ["a Texas b"])


def test_mean_ci():
    result = process.mean_ci([1, 0, 1, 1], n_boot=200, seed=42, name="x")

    assert result["x"] == 0.75
    assert result["x_low"] <= 0.75 <= result["x_high"]


def test_holm():
    adjusted = process.holm(np.array([0.01, 0.04, 0.03, 0.5]))

    np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.5])


def test_label_accuracy():
    data = read_csv("label_predictions.csv")

    result = process.label_accuracy(data["label"].to_numpy(), data["pred"].to_numpy(), n_labels=4)

    assert result.tolist() == pytest.approx([0.5, 1.0, 0.0, 0.6])  # le type 3 absent prend 0.6


def test_slice_gaps():
    data = read_csv("gap_inputs.csv")
    slices = data[["group"]]

    table = process.slice_gaps(
        data["correct"].to_numpy(float), data["expected"].to_numpy(float), slices, 1000, 0, 50, 0.05
    )

    status = dict(zip(table["value"], table["status"]))
    assert status == {"bad": "écart", "good": "ok", "tiny": "n insuffisant"}
    assert table.set_index("value").loc["bad", "gap"] == -1.0


def test_automation_by_slice(predictions):
    correct = (predictions["label"] == predictions["pred"]).to_numpy()

    result = process.automation_by_slice(
        predictions["confidence"].to_numpy(), correct, 0.8, read_csv("slices.csv")
    )

    pd.testing.assert_frame_equal(result, read_csv("expected_automation.csv"), check_dtype=False)


def test_slice_samples():
    texts = read_csv("texts.csv")["text"].tolist()

    samples = process.slice_samples(texts, read_csv("texts_slices.csv"), n=2, seed=0)

    assert list(samples.columns) == ["slice", "value", "text"]
    assert samples["value"].value_counts().to_dict() == {"a": 2, "b": 2}


def test_counterfactual_effect():
    data = read_csv("counterfactual.csv")

    effect = process.counterfactual_effect(
        data["label"].to_numpy(), data["before"].to_numpy(), data["after"].to_numpy(),
        data["conf_before"].to_numpy(), data["conf_after"].to_numpy(), ["a", "b", "c"], 100, 0,
    )  # fmt: skip

    assert effect["n"] == 3
    assert effect["flip_rate"] == pytest.approx(1 / 3)
    assert effect["accuracy_change"] == pytest.approx(-1 / 3)
    assert effect["confidence_change"] == pytest.approx(-0.1)
    assert effect["top_flips"] == "b -> c (1)"


def test_ood_summary(predictions):
    summary = process.ood_summary(
        predictions["label"].to_numpy(), predictions["pred"].to_numpy(),
        predictions["confidence"].to_numpy(), threshold=0.8, group=np.arange(3),
    )  # fmt: skip

    assert summary["n"] == 10
    assert summary["accuracy"] == pytest.approx(0.7)
    assert summary["mean_confidence"] == pytest.approx(0.704)
    assert summary["auto_rate"] == 0.5
    assert summary["auto_precision"] == 1.0


def test_cell():
    assert process.cell(0.12345) == "0.123"


def test_markdown_table():
    table = process.markdown_table(read_csv("table.csv"))

    assert table == (TEST_DATA / "expected_table.md").read_text()


def test_section():
    assert process.section("Titre", "a", "b") == "## Titre\n\na\n\nb"


def test_pick():
    table = process.pick(read_csv("columns.csv"), ["a", "c", "absente"])

    assert table == "| a | c |\n|---|---|\n| 1 | 3 |"


def test_summary_lines(report_inputs):
    lines = process.summary_lines(*report_inputs.values())

    text = "\n".join(lines)
    assert "1 écart(s) significatif(s) sur 2 tranches testables" in text
    assert "pronoun = feminine : -10.0 points" in text
    assert "plus que le témoin, significatif" in text
    assert "0.850 en anglais contre 0.600 en français" in text
    assert "4.5 % des clauses de test sont identiques" in text


def test_render_report(report_inputs):
    report = process.render_report(
        report_inputs["gaps"], read_csv("expected_automation.csv"), report_inputs["counterfactuals"],
        report_inputs["ood"], read_csv("ood_clauses.csv"), report_inputs["leakage"], None, 0.9,
        report_inputs["cfg"],
    )  # fmt: skip

    assert report.startswith("# Audit de biais et de robustesse")
    for title in ("## Constats", "## 1. Accuracy par tranche", "## 5. Fuite", "## Limites"):
        assert title in report
    assert "## 6. Retrieval" not in report  # pas de tableau de retrieval fourni
