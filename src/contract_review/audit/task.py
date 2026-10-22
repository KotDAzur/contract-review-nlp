"""Étape audit : biais, robustesse et hors distribution. Génère AUDIT.md dans le dossier du run."""

import json
import re
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

from contract_review.audit import process
from contract_review.classification.task import predict_all
from contract_review.data.process import group_ids, normalize
from contract_review.data.task import load_ledgar
from contract_review.evaluation.task import MODELS
from contract_review.utils import ROOT, read_yaml

# Chaque perturbation est comparée à un témoin qui change autant le texte sans le dépayser.
CONTROLS = {
    "jurisdiction: England and Wales": "jurisdiction: Texas (témoin)",
    "jurisdiction: France": "jurisdiction: Texas (témoin)",
    "jurisdiction: Germany": "jurisdiction: Texas (témoin)",
    "company form: GmbH": "company form: Ltd. (témoin)",
    "company form: S.A.S.": "company form: Ltd. (témoin)",
}


@dataclass
class Audit:
    texts: list[str]
    labels: np.ndarray
    test_pred: pd.DataFrame  # label, pred_baseline, pred, confidence (étape evaluation)
    names: list[str]
    group: np.ndarray  # groupe de synonymes de chaque label
    threshold: float  # seuil de confiance de production
    n_boot: int
    seed: int


def predict_batches(ctx, batches: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    """Un seul passage du modèle pour tous les groupes de textes."""
    predicted = predict_all(ctx, [text for texts in batches.values() for text in texts])
    result, start = {}, 0
    for name, texts in batches.items():
        result[name] = predicted.iloc[start : start + len(texts)].reset_index(drop=True)
        start += len(texts)
    return result


def translate(model_name: str, texts: list[str], batch_size: int = 16) -> list[str]:
    """Anglais -> français avec un modèle MarianMT (traduction automatique, donc imparfaite)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device).eval()
    translated = [""] * len(texts)
    order = np.argsort([len(text) for text in texts])  # longueurs proches : moins de padding
    for start in range(0, len(texts), batch_size):
        chosen = order[start : start + batch_size]
        batch = tokenizer(
            [texts[i] for i in chosen],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(**batch, max_new_tokens=512, num_beams=1)
        for i, text in zip(chosen, tokenizer.batch_decode(generated, skip_special_tokens=True)):
            translated[i] = text
    return translated


def slice_analysis(a: Audit, val_pred, slices, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Écarts d'accuracy par tranche (corrigés du mélange de types) et charge de revue humaine."""
    val_labels = val_pred["label"].to_numpy()
    gaps = []
    for model, column in MODELS.items():
        label_acc = process.label_accuracy(val_labels, val_pred[column].to_numpy(), len(a.names))
        correct = (a.test_pred[column].to_numpy() == a.labels).astype(float)
        table = process.slice_gaps(
            correct,
            label_acc[a.labels],
            slices,
            a.n_boot,
            a.seed,
            cfg["min_slice_size"],
            cfg["alpha"],
        )
        gaps.append(table.assign(model=model))
    correct = (a.test_pred["pred"].to_numpy() == a.labels).astype(float)
    automation = process.automation_by_slice(
        a.test_pred["confidence"].to_numpy(), correct, a.threshold, slices
    )
    return pd.concat(gaps), automation


def counterfactual_analysis(a: Audit, variants, predicted) -> pd.DataFrame:
    """Effet de chaque perturbation, pour les deux modèles, avec l'excès par rapport au témoin."""
    rows, flips = [], {}
    for name, (idx, _) in variants.items():
        for model, column in MODELS.items():
            before, after = a.test_pred[column].to_numpy()[idx], predicted[name][column].to_numpy()
            conf = (None, None)
            if model == "transformer":
                conf = (
                    a.test_pred["confidence"].to_numpy()[idx],
                    predicted[name]["confidence"].to_numpy(),
                )
            row = {"perturbation": name, "model": model}
            row |= process.counterfactual_effect(
                a.labels[idx], before, after, *conf, a.names, a.n_boot, a.seed
            )
            flips[(name, model)] = before != after
            if name in CONTROLS:
                excess = flips[(name, model)].astype(float) - flips[(CONTROLS[name], model)]
                row |= process.mean_ci(excess, a.n_boot, a.seed, "flip_excess")
            rows.append(row)

            if name == "jurisdiction: France":  # cas spécial : clauses de droit applicable
                governing = a.group[a.labels[idx]] == a.group[a.names.index("Governing Laws")]
                special = process.counterfactual_effect(
                    a.labels[idx][governing], before[governing], after[governing], *(
                        (None, None) if conf[0] is None else (conf[0][governing], conf[1][governing])
                    ), a.names, a.n_boot, a.seed,
                )  # fmt: skip
                rows.append(
                    {
                        "perturbation": "Governing Laws: US state -> France",
                        "model": model,
                        **special,
                    }
                )
    return pd.DataFrame(rows)


def out_of_distribution(a: Audit, manual, sample, predicted) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clauses françaises : écrites à la main (qualitatif) et traduites automatiquement (chiffré)."""
    manual_labels = np.array([a.names.index(item["label"]) for item in manual])
    rows, clauses = [], []
    for model, column in MODELS.items():
        sets = {
            "manuel EN": (manual_labels, predicted["ood_en"]),
            "manuel FR": (manual_labels, predicted["ood_fr"]),
            "test traduit, original EN": (
                a.labels[sample],
                a.test_pred.iloc[sample].reset_index(drop=True),
            ),
            "test traduit, FR": (a.labels[sample], predicted["translated"]),
        }
        for name, (labels, pred) in sets.items():
            summary = process.ood_summary(
                labels, pred[column].to_numpy(), pred["confidence"].to_numpy(), a.threshold, a.group
            )
            rows.append({"model": model, "set": name, **summary})
        drop = (sets["test traduit, FR"][1][column].to_numpy() == a.labels[sample]).astype(float)
        drop -= a.test_pred[column].to_numpy()[sample] == a.labels[sample]
        rows.append(
            {
                "model": model,
                "set": "test traduit, FR moins EN",
                **process.mean_ci(drop, a.n_boot, a.seed, "accuracy_change"),
            }
        )
    for lang in ("fr", "en"):
        pred = predicted[f"ood_{lang}"]
        clauses.append(
            pd.DataFrame(
                {
                    "lang": lang,
                    "expected": [item["label"] for item in manual],
                    "transformer": [a.names[i] for i in pred["pred"]],
                    "confidence": pred["confidence"],
                    "baseline": [a.names[i] for i in pred["pred_baseline"]],
                    "text": [item[lang][:90] for item in manual],
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(clauses)


def leakage_table(a: Audit, duplicate: np.ndarray) -> pd.DataFrame:
    """Accuracy avec et sans les clauses de test identiques à une clause du train."""
    rows = []
    for model, column in MODELS.items():
        correct = (a.test_pred[column].to_numpy() == a.labels).astype(float)
        rows.append(
            {
                "model": model,
                "duplicate_share": duplicate.mean(),
                "accuracy_all": correct.mean(),
                **process.mean_ci(
                    correct[~duplicate], a.n_boot, a.seed, "accuracy_without_duplicates"
                ),
            }
        )
    return pd.DataFrame(rows)


def run(ctx) -> None:
    cfg, seed = ctx.cfg["audit"], ctx.cfg["seed"]
    ledgar = load_ledgar(ctx.cfg["data"])
    names = ledgar["train"].features["label"].names
    texts = list(ledgar["test"]["text"])
    metrics_dir = ctx.dir("metrics")
    evaluation = json.loads((metrics_dir / "evaluation.json").read_text())
    a = Audit(
        texts=texts,
        labels=np.array(ledgar["test"]["label"]),
        test_pred=pd.read_parquet(metrics_dir / "predictions_test.parquet"),
        names=names,
        group=group_ids(names, ctx.cfg["synonym_groups"]),
        threshold=evaluation["transformer"]["selective"]["threshold"],
        n_boot=cfg["n_bootstrap"],
        seed=seed,
    )

    # Tranches, dont la fuite : clause de test identique à une clause du train (casse et ponctuation exclues)
    slices = pd.read_parquet(ctx.dir("data") / "slices_test.parquet")
    train_texts = {normalize(text) for text in ledgar["train"]["text"]}
    duplicate = np.array([normalize(text) in train_texts for text in texts])
    slices["duplicate_in_train"] = np.where(duplicate, "yes", "no")
    gaps, automation = slice_analysis(
        a, pd.read_parquet(metrics_dir / "predictions_validation.parquet"), slices, cfg
    )

    # Perturbations : les témoins viennent avant les perturbations qu'ils servent à comparer
    jurisdiction, _ = process.perturb(texts, lambda t: process.replace_jurisdiction(t, "Texas"))
    variants = {
        "gender: masculine -> feminine": process.perturb(
            texts, lambda t: process.swap_gender(t, "feminine")
        ),
        "gender: feminine -> masculine": process.perturb(
            texts, lambda t: process.swap_gender(t, "masculine")
        ),
        "gender: feminine -> masculine, sans 'her'": process.perturb(
            texts,
            lambda t: (
                t
                if re.search(r"\bher\b", t, re.IGNORECASE)
                else process.swap_gender(t, "masculine")
            ),
        ),
    }
    for place in ("Texas (témoin)", "England and Wales", "France", "Germany"):
        new = [
            process.replace_jurisdiction(texts[i], place.removesuffix(" (témoin)"))
            for i in jurisdiction
        ]
        variants[f"jurisdiction: {place}"] = (jurisdiction, new)
    for form in ("Ltd. (témoin)", "GmbH", "S.A.S."):
        change = partial(process.replace_company_form, form=form.removesuffix(" (témoin)"))
        variants[f"company form: {form}"] = process.perturb(texts, change)

    # Hors distribution : 12 clauses écrites à la main, et un échantillon du test traduit
    manual = read_yaml(ROOT / cfg["ood_file"])
    words = np.array([len(text.split()) for text in texts])
    sample = np.sort(
        np.random.default_rng(seed).choice(
            np.flatnonzero(words <= 200), cfg["translated_clauses"], replace=False
        )
    )
    translated = translate(cfg["translation_model"], [texts[i] for i in sample])

    batches = {name: new for name, (_, new) in variants.items()}
    batches |= {
        "ood_fr": [item["fr"] for item in manual],
        "ood_en": [item["en"] for item in manual],
        "translated": translated,
    }
    predicted = predict_batches(ctx, batches)

    counterfactuals = counterfactual_analysis(a, variants, predicted)
    ood, ood_clauses = out_of_distribution(a, manual, sample, predicted)
    leakage = leakage_table(a, duplicate)

    retrieval_path = metrics_dir / "retrieval_slices.csv"
    retrieval = pd.read_csv(retrieval_path) if retrieval_path.exists() else None
    gaps.to_csv(metrics_dir / "audit_slices.csv", index=False)
    automation.to_csv(metrics_dir / "audit_automation.csv", index=False)
    counterfactuals.to_csv(metrics_dir / "audit_counterfactuals.csv", index=False)
    ood.to_csv(metrics_dir / "audit_ood.csv", index=False)
    ood_clauses.to_csv(metrics_dir / "audit_ood_clauses.csv", index=False)
    leakage.to_csv(metrics_dir / "audit_leakage.csv", index=False)
    process.slice_samples(texts, slices, cfg["samples_per_slice"], seed).to_csv(
        metrics_dir / "slice_samples.csv", index=False
    )
    report = process.render_report(
        gaps, automation, counterfactuals, ood, ood_clauses, leakage, retrieval, a.threshold, cfg
    )
    (ctx.run_dir / "AUDIT.md").write_text(report, encoding="utf-8")
    ctx.log.info("audit : %d écarts de tranche signalés", (gaps["status"] == "écart").sum())
