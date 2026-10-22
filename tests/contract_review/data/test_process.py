import re
from pathlib import Path

import numpy as np
import pandas as pd

from contract_review.data import process

TEST_DATA = Path(__file__).parent / "test_data"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TEST_DATA / name, keep_default_na=False)


def test_matches():
    found = process.matches("He said she would pay", process.PRONOUNS, re.IGNORECASE)

    assert found == ["masculine", "feminine"]


def test_slices_of():
    clauses, expected = read_csv("clauses.csv"), read_csv("expected_slices.csv")

    result = process.slices_of(clauses["text"][0], label_count=1500)

    assert result == expected.iloc[0].to_dict()


def test_compute_slices():
    clauses, counts = read_csv("clauses.csv"), read_csv("label_counts.csv")

    result = process.compute_slices(clauses["text"], clauses["label"], counts["count"].to_numpy())

    pd.testing.assert_frame_equal(result, read_csv("expected_slices.csv"))


def test_normalize():
    assert process.normalize("  The Agreement,  SHALL apply!! ") == "the agreement shall apply"


def test_group_ids():
    ids = process.group_ids(["a", "b", "c", "d"], [["b", "c"]])

    assert ids.tolist() == [0, 1, 1, 3]


def test_label_summary():
    summary = process.label_summary(["a", "b", "c"], np.array([1500, 400, 10]))

    assert summary["n_labels"] == 3
    assert summary["most_frequent"] == {"label": "a", "count": 1500}
    assert summary["least_frequent"] == {"label": "c", "count": 10}
    assert summary["imbalance_ratio"] == 150.0


def test_length_stats():
    stats = process.length_stats(read_csv("clauses.csv")["text"])

    assert stats == {"mean": 11.3, "median": 11.0, "p95": 12.8, "max": 13}
