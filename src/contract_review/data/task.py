"""Étape data : charge LEDGAR (révision épinglée), écrit statistiques et tranches du run."""

import json

import numpy as np
from datasets import load_dataset

from contract_review.data import process


def load_ledgar(cfg: dict):
    """Même fonction pour toutes les étapes : le dataset est figé par `revision`."""
    return load_dataset(cfg["dataset"], cfg["subset"], revision=cfg["revision"])


def run(ctx) -> None:
    ledgar = load_ledgar(ctx.cfg["data"])
    names = ledgar["train"].features["label"].names
    counts = np.bincount(ledgar["train"]["label"], minlength=len(names))
    # Tranches alignées ligne à ligne sur chaque split ; l'audit les relit.
    slices = {
        split: process.compute_slices(ds["text"], ds["label"], counts)
        for split, ds in ledgar.items()
    }

    stats = {
        "sizes": {split: len(ds) for split, ds in ledgar.items()},
        "labels": process.label_summary(names, counts),
        "length_words": {split: process.length_stats(ds["text"]) for split, ds in ledgar.items()},
        "slices": {
            split: {col: df[col].value_counts().to_dict() for col in df}
            for split, df in slices.items()
        },
    }
    out = ctx.dir("data")
    for split, df in slices.items():
        df.to_parquet(out / f"slices_{split}.parquet")
    (out / "label_names.json").write_text(json.dumps(names, indent=2))
    (out / "label_counts_train.json").write_text(
        json.dumps(dict(zip(names, counts.tolist())), indent=2)
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    ctx.log.info("data : %s | %s", stats["sizes"], stats["labels"])
