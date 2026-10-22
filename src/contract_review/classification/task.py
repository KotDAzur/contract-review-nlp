"""Étape classification : baseline TF-IDF puis fine-tuning du transformer."""

import json

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from contract_review.classification import process
from contract_review.data.task import load_ledgar


def predict_probs(ctx, texts) -> np.ndarray:
    """Probabilités par type de clause, avec le classifieur entraîné par l'étape classification."""
    cfg = ctx.cfg["classification"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = process.load_classifier(ctx.dir("models") / "classifier", device)
    loader = process.make_loader(tokenizer, texts, None, cfg["max_len"], cfg["eval_batch_size"])
    use_amp = cfg["fp16"] and device.type == "cuda"
    return process.softmax(process.predict_logits(model, loader, device, use_amp))


def run(ctx) -> None:
    cfg = ctx.cfg["classification"]
    ledgar = load_ledgar(ctx.cfg["data"])
    train, val = ledgar["train"], ledgar["validation"]
    names = train.features["label"].names
    models = ctx.dir("models")

    baseline = process.make_baseline().fit(train["text"], train["label"])
    joblib.dump(baseline, models / "baseline.joblib")
    ctx.log.info(
        "baseline TF-IDF, validation : %s",
        process.scores(val["label"], baseline.predict(val["text"])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=len(names), id2label=dict(enumerate(names))
    )
    train_loader = process.make_loader(
        tokenizer, train["text"], train["label"], cfg["max_len"], cfg["batch_size"],
        shuffle=True, seed=ctx.cfg["seed"],
    )  # fmt: skip
    val_loader = process.make_loader(
        tokenizer, val["text"], val["label"], cfg["max_len"], cfg["eval_batch_size"]
    )
    counts = np.bincount(train["label"], minlength=len(names))
    weights = process.class_weights(counts, cfg["class_weights"])

    best_state, history = process.train_model(
        model, train_loader, val_loader, val["label"], cfg, weights, device, ctx.log
    )
    model.load_state_dict(best_state)
    model.save_pretrained(models / "classifier")
    tokenizer.save_pretrained(models / "classifier")
    (ctx.dir("metrics") / "history.json").write_text(json.dumps(history, indent=2))
