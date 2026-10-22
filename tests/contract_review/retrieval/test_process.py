from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contract_review.data.process import normalize
from contract_review.retrieval import process

TEST_DATA = Path(__file__).parent / "test_data"
BANK_LABELS = np.array([0, 1, 0, 1])


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TEST_DATA / name, keep_default_na=False)


@pytest.fixture
def bank_texts():
    return read_csv("bank.csv")["text"].tolist()


@pytest.fixture
def index():
    return process.build_index(read_csv("embeddings.csv").to_numpy(dtype="float32"))


def test_tokenize():
    assert process.tokenize("Thirty (30) Days, notice!") == ["thirty", "30", "days", "notice"]


def test_build_bm25(bank_texts):
    assert process.build_bm25(bank_texts).corpus_size == 4


def test_bm25_search(bank_texts):
    bm25 = process.build_bm25(bank_texts)

    found = process.bm25_search(bm25, "terminate thirty days notice", k=1)

    assert found.tolist() == [0]  # "thirty" n'apparaît que dans la clause 0


def test_encode(tiny_embedder):
    embeddings = process.encode(tiny_embedder, ["the agreement", "notice days"])

    assert embeddings.shape == (2, 16)
    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_build_index(index):
    assert index.ntotal == 4


def test_search(index):
    query = np.array([1.0, 0.0], dtype="float32")

    everywhere = process.search(index, query, k=3)
    restricted = process.search(index, query, k=3, allowed=np.array([2, 3], dtype=np.int64))

    assert everywhere == [0, 1, 2]
    assert restricted == [2, 3]


def test_extract_constraints():
    constraints = process.extract_constraints("termination with 30 days notice in Delaware")

    assert constraints == [r"(?i)\b(30|thirty)\b.{0,12}\bdays?\b", r"\bDelaware\b"]


def test_question_search(index, bank_texts):
    query = np.array([1.0, 0.0], dtype="float32")
    label_embs = read_csv("label_embeddings.csv").to_numpy(dtype="float32")
    constraints = process.extract_constraints("30 days notice")

    found = process.question_search(
        index, query, label_embs, BANK_LABELS, bank_texts, constraints, k=3
    )

    assert found == [0]  # seule la clause avec "thirty (30) days" respecte la contrainte


def test_rank():
    copies_of = {normalize("The query text"): [0]}

    ranked = process.rank(
        lambda i, n: [0, 1, 2, 3][:n], ["The query, text!"], k=2, copies_of=copies_of
    )

    assert ranked == [[1, 2]]  # la copie exacte (id 0) est écartée


def test_relevance():
    ranked, query_labels = [[1, 2], [1, 2]], [1, 0]

    result = process.relevance(ranked, query_labels, BANK_LABELS, np.arange(2), k=3)

    np.testing.assert_array_equal(result, read_csv("expected_relevance.csv").to_numpy())


def test_n_relevant():
    copies_of = {normalize("the query text"): [0]}

    result = process.n_relevant(
        ["the query text", "other"], [0, 1], BANK_LABELS, copies_of, np.arange(2)
    )

    assert result.tolist() == [1, 2]  # 2 clauses du type 0, dont la copie exclue


def test_retrieval_metrics():
    rel = read_csv("rel_metrics.csv").to_numpy(dtype=float)

    result = process.retrieval_metrics(rel, np.array([2, 1]))

    assert result["p@1"] == 0.5
    assert result["p@5"] == pytest.approx(0.375)
    assert result["p@10"] == pytest.approx(0.375)
    assert result["mrr@10"] == pytest.approx(0.75)
    assert result["ndcg@10"] == pytest.approx(0.815465, abs=1e-5)


def test_slice_precision():
    rel = read_csv("rel_slices.csv").to_numpy(dtype=float)

    result = process.slice_precision(rel, read_csv("slices.csv"))

    pd.testing.assert_frame_equal(
        result, read_csv("expected_slice_precision.csv"), check_dtype=False
    )


def test_question_scores(bank_texts):
    questions = read_csv("questions.csv").to_dict("records")
    strict = read_csv("question_rel_strict.csv").to_numpy(dtype=float)
    tolerant = read_csv("question_rel_tolerant.csv").to_numpy(dtype=float)

    scores = process.question_scores([[0, 1], [2]], questions, bank_texts, strict, tolerant)

    assert scores["p@10"].tolist() == [0.1, 0.0]
    assert scores["p@10_tolerant"].tolist() == [0.2, 0.0]
    assert scores["constraint_hit@10"][0] == 0.1  # un résultat sur dix contient "thirty"
    assert np.isnan(scores["constraint_hit@10"][1])  # pas de contrainte annotée
