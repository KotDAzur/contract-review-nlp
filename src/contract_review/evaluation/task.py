"""Étape evaluation : évalue baseline et transformer. Le test n'est utilisé qu'ici."""

import json

import joblib
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from contract_review.classification import process as classification
from contract_review.data.task import load_ledgar
from contract_review.evaluation import process

MODELS = {"baseline": "pred_baseline", "transformer": "pred"}  # nom -> colonne de prédictions


def predict(ctx, ledgar) -> dict[str, pd.DataFrame]:
    """Prédictions des deux modèles sur la validation et le test."""
    cfg = ctx.cfg["classification"]
    models = ctx.dir("models")
    baseline = joblib.load(models / "baseline.joblib")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(models / "classifier")
    model = AutoModelForSequenceClassification.from_pretrained(models / "classifier").to(device)

    predictions = {}
    for split in ("validation", "test"):
        texts, labels = list(ledgar[split]["text"]), list(ledgar[split]["label"])
        loader = classification.make_loader(
            tokenizer, texts, labels, cfg["max_len"], cfg["eval_batch_size"]
        )
        logits = classification.predict_logits(
            model, loader, device, use_amp=cfg["fp16"] and device.type == "cuda"
        )
        probs = process.softmax(logits)
        predictions[split] = pd.DataFrame(
            {
                "label": labels,
                "pred_baseline": baseline.predict(texts),
                "pred": probs.argmax(axis=1),
                "confidence": probs.max(axis=1),
            }
        )
    return predictions


def score_models(val, test, ids, n_boot, seed) -> dict:
    """Métriques de chaque modèle (IC bootstrap sur le test) et écart transformer - baseline."""
    labels = test["label"].to_numpy()
    samples = {
        model: process.bootstrap_metrics(labels, test[column], n_boot, seed)
        for model, column in MODELS.items()
    }
    results = {}
    for model, column in MODELS.items():
        validation = process.metrics(val["label"], val[column])
        results[model] = {
            "validation": {name: round(value, 4) for name, value in validation.items()},
            "test": process.with_ci(process.metrics(labels, test[column]), samples[model]),
            "test_tolerant_accuracy": round(
                process.tolerant_accuracy(labels, test[column].to_numpy(), ids), 4
            ),
        }
    gain = {
        name: value["value"] - results["baseline"]["test"][name]["value"]
        for name, value in results["transformer"]["test"].items()
    }
    results["transformer_minus_baseline"] = process.with_ci(
        gain, samples["transformer"] - samples["baseline"]
    )
    return results


def confidence_and_correct(df: pd.DataFrame):
    return df["confidence"].to_numpy(), (df["label"] == df["pred"]).to_numpy()


def run(ctx) -> None:
    cfg = ctx.cfg["evaluation"]
    ledgar = load_ledgar(ctx.cfg["data"])
    names = ledgar["train"].features["label"].names
    ids = process.group_ids(names, ctx.cfg["synonym_groups"])

    predictions = predict(ctx, ledgar)
    val, test = predictions["validation"], predictions["test"]
    labels, preds = test["label"].to_numpy(), test["pred"].to_numpy()
    results = score_models(val, test, ids, cfg["n_bootstrap"], ctx.cfg["seed"])

    # Seuil de confiance : choisi sur la validation, vérifié sur le test
    conf_val, correct_val = confidence_and_correct(val)
    conf_test, correct_test = confidence_and_correct(test)
    curve_test = process.coverage_precision(conf_test, correct_test)
    threshold = process.choose_threshold(
        process.coverage_precision(conf_val, correct_val), cfg["target_precision"]
    )
    on_test = process.at_threshold(conf_test, correct_test, threshold)
    calibration, ece = process.reliability(conf_test, correct_test, cfg["ece_bins"])
    results["transformer"]["ece"] = round(ece, 4)
    results["transformer"]["selective"] = {
        "target_precision": cfg["target_precision"],
        "threshold": round(threshold, 4),
        "validation": process.at_threshold(conf_val, correct_val, threshold),
        "test": on_test,
    }

    # Détail par type de clause, confusions et tranches
    classes = process.per_class(labels, preds, names)
    classes["f1_baseline"] = process.per_class(labels, test["pred_baseline"], names)["f1"]
    slices = pd.read_parquet(ctx.dir("data") / "slices_test.parquet")
    slice_tables = {
        model: process.slice_accuracy(
            (test[column] == test["label"]).to_numpy(), slices, cfg["n_bootstrap"], ctx.cfg["seed"]
        )
        for model, column in MODELS.items()
    }

    metrics_dir, figures_dir = ctx.dir("metrics"), ctx.dir("figures")
    (metrics_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
    classes.to_csv(metrics_dir / "per_class.csv", index=False)
    process.top_confusions(labels, preds, names).to_csv(metrics_dir / "confusions.csv", index=False)
    slices_table = pd.concat(slice_tables, names=["model"]).reset_index(level=0)
    slices_table.to_csv(metrics_dir / "slices.csv", index=False)
    for split, df in predictions.items():
        df.to_parquet(metrics_dir / f"predictions_{split}.parquet")

    process.plot_weakest(classes).savefig(figures_dir / "weakest_classes.png", dpi=150)
    process.plot_reliability(calibration, ece).savefig(figures_dir / "reliability.png", dpi=150)
    process.plot_coverage(curve_test, cfg["target_precision"], on_test).savefig(
        figures_dir / "coverage_precision.png", dpi=150
    )
    scores = {model: results[model]["test"]["macro_f1"]["value"] for model in MODELS}
    ctx.log.info("evaluation, macro-F1 test : %s", scores)
