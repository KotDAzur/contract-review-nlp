import re
from collections import Counter

import numpy as np
import pandas as pd

from contract_review.data.process import PRONOUNS, US_STATES, matches

MASC_TO_FEM = {"he": "she", "him": "her", "himself": "herself", "his": "her"}
FEM_TO_MASC = {"she": "he", "herself": "himself", "hers": "his"}
# "her" peut devenir "him" (objet) ou "his" (possessif) : on regarde le mot qui suit.
OBJECT_CUES = {"and", "or", "to", "for", "from", "with", "by", "at", "in", "on", "of", "as", "that"}
STATE_RE = re.compile(
    rf"\b(?:[Tt]he )?(?:State|Commonwealth) of (?:{US_STATES})\b|\b(?:{US_STATES})\b"
)
COMPANY_RE = re.compile(r"(?<!\w)(?:Inc\.|Corp\.|L\.L\.C\.|LLC|Corporation)(?!\w)")


# ----- Perturbations contrefactuelles : une seule chose change dans la clause -----


def match_case(word: str, model: str) -> str:
    if model.isupper() and len(model) > 1:
        return word.upper()
    return word.capitalize() if model[0].isupper() else word


def swap_gender(text: str, to: str) -> str:
    """he/him/his vers she/her (to="feminine"), ou l'inverse (to="masculine").

    Une clause qui mélange déjà les deux genres ("his or her") reste inchangée : la perturbation
    donnerait "his or his", un texte absurde qui ne mesurerait plus la sensibilité au genre."""
    if len(matches(text, PRONOUNS, re.IGNORECASE)) == 2:
        return text
    to_feminine = to == "feminine"
    words = MASC_TO_FEM if to_feminine else FEM_TO_MASC
    pattern = r"\b(he|him|himself|his)\b" if to_feminine else r"\b(she|her|hers|herself)\b"

    def replace(match: re.Match) -> str:
        word = match.group(0)
        if word.lower() == "her":
            following = re.match(r"\s*(\w*)", text[match.end() :]).group(1).lower()
            new = "him" if not following or following in OBJECT_CUES else "his"
        else:
            new = words[word.lower()]
        return match_case(new, word)

    return re.sub(pattern, replace, text, flags=re.IGNORECASE)


def replace_jurisdiction(text: str, place: str) -> str:
    """Remplace chaque État américain ("the State of New York" compris) par `place`."""
    return STATE_RE.sub(place, text)


def replace_company_form(text: str, form: str) -> str:
    return COMPANY_RE.sub(form, text)


def perturb(texts: list[str], change) -> tuple[list[int], list[str]]:
    """Indices et nouveaux textes des seules clauses que `change` modifie."""
    changed = [(i, new) for i, text in enumerate(texts) if (new := change(text)) != text]
    return [i for i, _ in changed], [new for _, new in changed]


# ----- Statistiques -----


def mean_ci(values, n_boot: int, seed: int, name: str) -> dict[str, float]:
    """Moyenne et IC à 95 % par bootstrap. Même seed et même taille = rééchantillons appariés."""
    values = np.asarray(values, dtype=float)
    resamples = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    boot = values[resamples].mean(axis=1)
    return {
        name: values.mean(),
        f"{name}_low": np.percentile(boot, 2.5),
        f"{name}_high": np.percentile(boot, 97.5),
    }


def holm(p_values: np.ndarray) -> np.ndarray:
    """Correction de Holm : contrôle le risque de faux positif sur l'ensemble des tests."""
    order = np.argsort(p_values)
    factors = len(p_values) - np.arange(len(p_values))
    adjusted = np.minimum(1, np.maximum.accumulate(factors * p_values[order]))
    result = np.empty(len(p_values))
    result[order] = adjusted
    return result


def label_accuracy(labels, preds, n_labels: int) -> np.ndarray:
    """Accuracy par type de clause (accuracy globale si le type est absent)."""
    hits = np.bincount(labels[preds == labels], minlength=n_labels)
    total = np.bincount(labels, minlength=n_labels)
    return np.where(total > 0, hits / np.maximum(total, 1), (preds == labels).mean())


# ----- Tranches -----


def slice_gaps(correct, expected, slices, n_boot, seed, min_size, alpha) -> pd.DataFrame:
    """Écart entre l'accuracy d'une tranche et celle attendue vu son mélange de types de clause.

    `expected` : pour chaque clause, l'accuracy de son type mesurée sur le test lui-même. Sans cette
    correction, une tranche riche en types difficiles passerait pour un biais. Tirée de la
    validation, elle ajouterait un bruit que l'intervalle ignore et un décalage (le checkpoint a
    été choisi sur la validation) : des écarts significatifs seraient signalés à tort."""
    rng = np.random.default_rng(seed)
    rows = []
    for column in slices:
        for value, idx in slices.groupby(column).indices.items():
            diff = correct[idx] - expected[idx]
            boot = diff[rng.integers(0, len(idx), size=(n_boot, len(idx)))].mean(axis=1)
            below, above = (boot <= 0).sum() + 1, (boot >= 0).sum() + 1
            rows.append(
                {
                    "slice": column,
                    "value": value,
                    "n": len(idx),
                    "accuracy": correct[idx].mean(),
                    "expected": expected[idx].mean(),
                    "gap": diff.mean(),
                    "ci_low": np.percentile(boot, 2.5),
                    "ci_high": np.percentile(boot, 97.5),
                    "p": min(1.0, 2 * min(below, above) / (n_boot + 1)),
                }
            )
    table = pd.DataFrame(rows)
    testable = table["n"] >= min_size
    table["p_adj"] = np.nan
    table.loc[testable, "p_adj"] = holm(table.loc[testable, "p"].to_numpy())
    table["status"] = np.where(
        ~testable, "n insuffisant", np.where(table["p_adj"] < alpha, "écart", "ok")
    )
    return table.drop(columns="p")


def automation_by_slice(confidence, correct, threshold, slices) -> pd.DataFrame:
    """Charge de revue humaine : part étiquetée automatiquement et sa précision, par tranche."""
    auto = confidence >= threshold
    rows = [
        {
            "slice": column,
            "value": value,
            "n": len(idx),
            "auto_rate": auto[idx].mean(),
            "auto_precision": correct[idx][auto[idx]].mean() if auto[idx].any() else np.nan,
        }
        for column in slices
        for value, idx in slices.groupby(column).indices.items()
    ]
    return pd.DataFrame(rows)


def slice_samples(texts, slices, n: int, seed: int) -> pd.DataFrame:
    """Échantillon de clauses par tranche, à relire pour juger la qualité des regex."""
    data = slices.assign(text=[t[:300] for t in texts])
    groups = [
        data[data[column] == value]
        .sample(min(n, (data[column] == value).sum()), random_state=seed)
        .assign(slice=column, value=value)[["slice", "value", "text"]]
        for column in slices
        for value in data[column].unique()
    ]
    return pd.concat(groups)


# ----- Effets des perturbations et hors distribution -----


def counterfactual_effect(labels, before, after, conf_before, conf_after, names, n_boot, seed):
    """Effet d'une perturbation sur les clauses modifiées. Les confiances sont facultatives."""
    flipped = before != after
    moves = Counter(zip(np.array(names)[before[flipped]], np.array(names)[after[flipped]]))
    effect = {"n": len(labels)}
    effect |= mean_ci(flipped, n_boot, seed, "flip_rate")
    effect |= mean_ci(
        (after == labels).astype(float) - (before == labels), n_boot, seed, "accuracy_change"
    )
    if conf_before is not None:
        effect |= mean_ci(conf_after - conf_before, n_boot, seed, "confidence_change")
    effect["top_flips"] = "; ".join(f"{a} -> {b} ({n})" for (a, b), n in moves.most_common(3))
    return effect


def ood_summary(labels, preds, confidence, threshold, group) -> dict:
    """Précision et part étiquetée automatiquement au seuil de production."""
    auto, correct = confidence >= threshold, preds == labels
    return {
        "n": len(labels),
        "accuracy": correct.mean(),
        "accuracy_tolerant": (group[preds] == group[labels]).mean(),
        "mean_confidence": confidence.mean(),
        "auto_rate": auto.mean(),
        "auto_precision": correct[auto].mean() if auto.any() else np.nan,
    }


# ----- Rapport -----


def cell(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "-" if np.isnan(value) else f"{value:.3f}"
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


LIMITS = """- Les tranches sont déduites du texte par regex : ce sont des heuristiques, pas des attributs
  réels (voir `slice_samples.csv` pour relire des exemples). LEDGAR n'a aucune métadonnée sur
  les parties ni sur le type de contrat.
- Le test de genre ne remplace que des pronoms. Il ne dit rien de biais plus subtils, et "her"
  (him ou his) est résolu par une heuristique.
- Un écart non significatif ne prouve pas l'absence de biais : les petites tranches ont des
  intervalles larges, et celles de moins de n clauses ne sont pas testées.
- Les clauses françaises traduites automatiquement mesurent la langue, pas le droit civil
  européen, dont le vocabulaire et la structure diffèrent. Cet audit ne permet pas d'affirmer
  que l'outil fonctionne pour des clients européens : seul un pilote annoté le montrerait.
- Un seul entraînement (seed 42) : la variabilité entre entraînements n'est pas mesurée."""


def section(title: str, *parts: str) -> str:
    return f"## {title}\n\n" + "\n\n".join(parts)


def pick(df: pd.DataFrame, columns: list[str]) -> str:
    return markdown_table(df[[c for c in columns if c in df.columns]])


def summary_lines(gaps, counterfactuals, ood, leakage, cfg) -> list[str]:
    """Constats chiffrés, générés à partir des tableaux."""
    transformer = gaps[gaps["model"] == "transformer"]
    testable = (transformer["status"] != "n insuffisant").sum()
    flagged = transformer[transformer["status"] == "écart"]
    lines = [
        (
            f"- Tranches : {len(flagged)} écart(s) significatif(s) sur {testable} tranches testables "
            f"(correction de Holm, seuil {cfg['alpha']}, corrigé du mélange de types de clause)."
        )
    ]
    lines += [
        f"  - {r.slice} = {r.value} : {100 * r.gap:+.1f} points par rapport à l'attendu (n = {r.n})"
        for r in flagged.itertuples()
    ]
    for r in counterfactuals[counterfactuals["model"] == "transformer"].itertuples():
        excess = ""
        if not np.isnan(getattr(r, "flip_excess", np.nan)):
            if r.flip_excess_low > 0:
                verdict = "plus que le témoin, significatif"
            elif r.flip_excess_high < 0:
                verdict = "moins que le témoin, significatif"
            else:
                verdict = "pas de différence significative"
            excess = f", écart au témoin {100 * r.flip_excess:+.1f} points ({verdict})"
        lines.append(
            f"- {r.perturbation} : {100 * r.flip_rate:.1f} % de prédictions changées (n = {r.n}){excess}."
        )
    fr = ood[(ood["model"] == "transformer") & (ood["set"] == "test traduit, FR")].iloc[0]
    en = ood[(ood["model"] == "transformer") & (ood["set"] == "test traduit, original EN")].iloc[0]
    precision = "sans objet" if np.isnan(fr["auto_precision"]) else f"{fr['auto_precision']:.3f}"
    lines.append(
        f"- Français (test traduit, n = {fr['n']:.0f}) : accuracy {en['accuracy']:.3f} en anglais "
        f"contre {fr['accuracy']:.3f} en français ; {100 * fr['auto_rate']:.1f} % des clauses françaises "
        f"seraient étiquetées automatiquement, avec une précision de {precision}."
    )
    leak = leakage[leakage["model"] == "transformer"].iloc[0]
    lines.append(
        f"- Fuite : {100 * leak['duplicate_share']:.1f} % des clauses de test sont identiques à une clause du "
        f"train (casse et ponctuation exclues) ; accuracy {leak['accuracy_all']:.3f} avec elles, "
        f"{leak['accuracy_without_duplicates']:.3f} sans."
    )
    return lines


def render_report(
    gaps, automation, counterfactuals, ood, ood_clauses, leakage, retrieval, threshold, cfg
) -> str:
    """AUDIT.md : constats en tête, puis tous les tableaux."""
    slice_columns = [
        "slice",
        "value",
        "n",
        "accuracy",
        "expected",
        "gap",
        "ci_low",
        "ci_high",
        "p_adj",
        "status",
    ]
    transformer = gaps[gaps["model"] == "transformer"]
    baseline_flagged = gaps[(gaps["model"] == "baseline") & (gaps["status"] == "écart")]
    effect_columns = ["perturbation", "model", "n", "flip_rate", "flip_rate_low", "flip_rate_high"]
    effect_columns += [
        "flip_excess",
        "flip_excess_low",
        "flip_excess_high",
        "accuracy_change",
        "confidence_change",
        "top_flips",
    ]
    ood_columns = [
        "model",
        "set",
        "n",
        "accuracy",
        "accuracy_tolerant",
        "mean_confidence",
        "auto_rate",
        "auto_precision",
    ]
    ood_columns += ["accuracy_change", "accuracy_change_low", "accuracy_change_high"]
    sections = [
        "# Audit de biais et de robustesse",
        section("Constats", "\n".join(summary_lines(gaps, counterfactuals, ood, leakage, cfg))),
        section(
            "1. Accuracy par tranche (transformer)",
            "`expected` est l'accuracy attendue vu les types de clause de la tranche (accuracy de "
            "chaque type sur l'ensemble du test) ; `gap` = accuracy - expected, avec son intervalle "
            "de confiance bootstrap.",
            pick(transformer, slice_columns),
            "Écarts significatifs du baseline TF-IDF : "
            + (
                "aucun."
                if baseline_flagged.empty
                else "\n\n" + pick(baseline_flagged, slice_columns)
            ),
        ),
        section(
            "2. Charge de revue humaine par tranche (transformer)",
            f"Seuil de confiance de production : {threshold:.3f}. `auto_rate` = part étiquetée "
            "automatiquement ; le reste part en revue humaine.",
            markdown_table(automation),
        ),
        section(
            "3. Tests contrefactuels",
            "Seules les clauses modifiées par la perturbation sont comptées. `flip_excess` est "
            "l'écart de changements par rapport au témoin, sur les mêmes clauses.",
            pick(counterfactuals, effect_columns),
        ),
        section(
            "4. Hors distribution : français",
            "Les lignes `manuel` portent sur 12 clauses écrites à la main (qualitatif). Les lignes "
            "`test traduit` comparent les mêmes clauses de test en anglais et en français "
            "automatique.",
            pick(ood, ood_columns),
            "Détail des 12 clauses écrites à la main :",
            markdown_table(ood_clauses),
        ),
        section("5. Fuite entre train et test", markdown_table(leakage)),
    ]
    if retrieval is not None:
        pivot = retrieval.pivot_table(
            index=["slice", "value", "n"], columns="system", values="p@10"
        ).reset_index()
        sections.append(
            section(
                "6. Retrieval : P@10 selon la tranche de la clause requête", markdown_table(pivot)
            )
        )
    sections.append(section("Limites", LIMITS))
    return "\n\n".join(sections) + "\n"
