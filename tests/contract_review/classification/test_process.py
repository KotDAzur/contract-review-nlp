import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from contract_review.classification import process

TEST_DATA = Path(__file__).parent / "test_data"
CPU = torch.device("cpu")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TEST_DATA / name)


@pytest.fixture
def clauses():
    return pd.read_csv(TEST_DATA / "ledgar" / "train.csv")


@pytest.fixture
def classifier(tiny_bert):
    return process.load_classifier(tiny_bert, CPU)


def test_make_baseline(clauses):
    baseline = process.make_baseline().fit(clauses["text"], clauses["label"])

    assert baseline.predict(clauses["text"]).tolist() == clauses["label"].tolist()


def test_scores():
    data = read_csv("predictions.csv")

    result = process.scores(data["label"], data["pred"])

    assert result["accuracy"] == pytest.approx(0.7)
    assert result["macro_f1"] == pytest.approx(0.69841, abs=1e-4)


def test_class_weights():
    weights = process.class_weights(np.array([100, 25, 4]), "sqrt_inv")

    assert weights.tolist() == pytest.approx([0.375, 0.75, 1.875])


def test_load_classifier(classifier):
    tokenizer, model = classifier

    assert model.config.num_labels == 3
    assert tokenizer("the agreement")["input_ids"][0] == tokenizer.cls_token_id


def test_make_loader(classifier, clauses):
    tokenizer, _ = classifier

    loader = process.make_loader(tokenizer, clauses["text"], clauses["label"], 8, batch_size=4)
    batch = next(iter(loader))

    assert len(loader) == 2  # 6 clauses, batches de 4
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape[1] <= 8  # tronqué à max_len, padding au plus long du batch
    assert batch["labels"].tolist() == clauses["label"][:4].tolist()


def test_softmax():
    probs = process.softmax(read_csv("logits.csv").to_numpy())

    np.testing.assert_allclose(probs, read_csv("expected_softmax.csv").to_numpy(), atol=1e-6)


def test_make_optimizer(classifier):
    _, model = classifier

    optimizer = process.make_optimizer(model, lr=1e-3, weight_decay=0.01)

    decay, no_decay = optimizer.param_groups
    assert (decay["weight_decay"], no_decay["weight_decay"]) == (0.01, 0.0)
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))


def test_predict_logits(classifier, clauses):
    tokenizer, model = classifier
    loader = process.make_loader(tokenizer, clauses["text"], None, 8, batch_size=4)

    logits = process.predict_logits(model, loader, CPU)

    assert logits.shape == (6, 3)


def test_train_model(classifier, clauses):
    tokenizer, model = classifier
    cfg = {"fp16": False, "lr": 1e-3, "weight_decay": 0.01, "epochs": 2, "warmup_ratio": 0.1}
    cfg |= {"grad_clip": 1.0, "patience": 2}
    train = process.make_loader(tokenizer, clauses["text"], clauses["label"], 8, 2, shuffle=True)
    val = process.make_loader(tokenizer, clauses["text"], clauses["label"], 8, 4)

    best_state, history = process.train_model(
        model, train, val, clauses["label"], cfg, None, CPU, logging.getLogger("test")
    )

    assert [epoch["epoch"] for epoch in history] == [1, 2]
    assert np.isfinite([epoch["train_loss"] for epoch in history]).all()
    assert set(best_state) == set(model.state_dict())
