import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def metrics(labels, preds) -> dict[str, float]:
    accuracy = accuracy_score(labels, preds)
    return {
        "accuracy": accuracy,
        "micro_f1": accuracy,  # une étiquette par clause : micro-F1 = accuracy
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def bootstrap_metrics(labels, preds, n_boot: int, seed: int) -> pd.DataFrame:
    """Métriques sur n_boot rééchantillons du test. Même seed = mêmes rééchantillons
    pour tous les modèles, donc les différences entre modèles sont appariées."""
    labels, preds = np.asarray(labels), np.asarray(preds)
    resamples = np.random.default_rng(seed).integers(0, len(labels), size=(n_boot, len(labels)))
    return pd.DataFrame(metrics(labels[i], preds[i]) for i in resamples)


def with_ci(point: dict, samples: pd.DataFrame) -> dict:
    """Valeur et intervalle de confiance à 95 % (percentiles du bootstrap)."""
    low, high = samples.quantile(0.025), samples.quantile(0.975)
    return {
        name: {"value": round(float(v), 4), "ci": [round(low[name], 4), round(high[name], 4)]}
        for name, v in point.items()
    }


def tolerant_accuracy(labels, preds, ids: np.ndarray) -> float:
    """Accuracy qui accepte une confusion entre quasi-synonymes."""
    return float(np.mean(ids[labels] == ids[preds]))


def per_class(labels, preds, names: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(names)), zero_division=0
    )
    return pd.DataFrame(
        {"label": names, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )


def top_confusions(labels, preds, names: list[str], k: int = 20) -> pd.DataFrame:
    wrong = labels != preds
    pairs = pd.DataFrame(
        {"true": np.array(names)[labels[wrong]], "pred": np.array(names)[preds[wrong]]}
    )
    return pairs.value_counts().head(k).rename("count").reset_index()


def reliability(confidence, correct, n_bins: int) -> tuple[pd.DataFrame, float]:
    """Accuracy vs confiance par intervalle, et ECE (écart moyen pondéré)."""
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows = [
        (confidence[bins == b].mean(), correct[bins == b].mean(), (bins == b).sum())
        for b in range(n_bins)
        if (bins == b).any()
    ]
    table = pd.DataFrame(rows, columns=["confidence", "accuracy", "count"])
    gap = (table["accuracy"] - table["confidence"]).abs()
    return table, float((table["count"] * gap).sum() / len(confidence))


def coverage_precision(confidence, correct) -> pd.DataFrame:
    """Si on ne garde que les clauses les plus sûres : part conservée et précision.

    Une ligne par valeur de confiance, la dernière du bloc : `confidence >= seuil` garde tout le
    bloc d'ex æquo, donc la précision à lire est celle qui le compte en entier."""
    order = np.argsort(-confidence)
    kept = np.arange(1, len(confidence) + 1)
    curve = pd.DataFrame(
        {
            "threshold": confidence[order],
            "coverage": kept / len(confidence),
            "precision": np.cumsum(correct[order]) / kept,
        }
    )
    return curve.drop_duplicates("threshold", keep="last")


def choose_threshold(curve: pd.DataFrame, target: float) -> float:
    """Seuil le plus bas qui atteint la précision visée.

    Si aucun seuil ne l'atteint : 2.0, au-dessus de toute confiance (maximum 1.0), donc plus
    aucune clause n'est étiquetée automatiquement. Un seuil de 1.0 garderait les clauses
    saturées à exactement 1.0."""
    reached = curve[curve["precision"] >= target]
    return float(reached["threshold"].iloc[-1]) if len(reached) else 2.0


def at_threshold(confidence, correct, threshold: float) -> dict:
    kept = confidence >= threshold
    return {
        "coverage": round(float(kept.mean()), 4),
        "precision": round(float(correct[kept].mean()), 4) if kept.any() else None,
    }


def slice_accuracy(correct, slices: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Accuracy et IC bootstrap pour chaque modalité de chaque tranche."""
    rng = np.random.default_rng(seed)
    rows = []
    for column in slices:
        for value, idx in slices.groupby(column).indices.items():
            hits = correct[idx]
            boot = hits[rng.integers(0, len(hits), size=(n_boot, len(hits)))].mean(axis=1)
            rows.append(
                {
                    "slice": column,
                    "value": value,
                    "n": len(hits),
                    "accuracy": hits.mean(),
                    "ci_low": np.percentile(boot, 2.5),
                    "ci_high": np.percentile(boot, 97.5),
                }
            )
    return pd.DataFrame(rows)


def plot_weakest(table: pd.DataFrame, k: int = 15) -> Figure:
    worst = table[table["support"] > 0].nsmallest(k, "f1")
    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    ax.barh(worst["label"], worst["f1"])
    ax.invert_yaxis()
    ax.set_xlabel("F1 sur le test")
    ax.set_title(f"Les {k} types de clause les plus faibles")
    fig.tight_layout()
    return fig


def plot_reliability(table: pd.DataFrame, ece: float) -> Figure:
    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], "--", color="gray", label="calibration parfaite")
    ax.plot(table["confidence"], table["accuracy"], "o-", label=f"modèle (ECE {ece:.3f})")
    ax.set_xlabel("confiance")
    ax.set_ylabel("accuracy")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_coverage(curve: pd.DataFrame, target: float, operating_point: dict) -> Figure:
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.plot(curve["coverage"], curve["precision"], label="test")
    ax.axhline(target, ls="--", color="gray", label=f"précision visée {target:.0%}")
    if operating_point["precision"] is not None:
        ax.plot(
            operating_point["coverage"],
            operating_point["precision"],
            "o",
            color="red",
            label="seuil choisi sur la validation",
        )
    ax.set_xlabel("part des clauses étiquetées automatiquement")
    ax.set_ylabel("précision")
    ax.legend()
    fig.tight_layout()
    return fig
