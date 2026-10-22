import re

import numpy as np
import pandas as pd

# LEDGAR n'a aucune métadonnée sur le contrat ou les parties : les tranches sont
# déduites du texte par regex. Ce sont des heuristiques, pas une vérité terrain.
US_STATES = (
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|"
    "Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|"
    "Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|"
    "New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|"
    "Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
    "West Virginia|Wisconsin|Wyoming"
)
NON_US = (
    "England|Wales|United Kingdom|France|Germany|Canada|Ontario|Ireland|Singapore|Hong Kong|"
    "Cayman|Bermuda|Netherlands|Switzerland|Luxembourg|Japan|China|India|Australia"
)

JURISDICTIONS = {"us_state": rf"\b({US_STATES})\b", "non_us": rf"\b({NON_US})\b"}
PRONOUNS = {"masculine": r"\b(he|him|his|himself)\b", "feminine": r"\b(she|her|hers|herself)\b"}
PARTIES = {
    "individual": r"\b(Employee|Executive|Consultant|Participant|Optionee|Grantee|Indemnitee)\b",
    "entity": r"\b(Company|Borrower|Lender|Corporation|Issuer|Trustee|Purchaser|Seller|Bank)\b",
}
# Ordre = priorité : la première famille qui matche l'emporte.
FAMILIES = {
    "employment": r"\b(employee|employment|executive|severance)\b",
    "lease": r"\b(lease|lessor|lessee|landlord|tenant)\b",
    "license": r"\b(license|licensor|licensee|royalt\w+)\b",
    "m_and_a": r"\b(merger|acquisition|purchase agreement|seller|purchaser)\b",
    "credit": r"\b(borrower|lender|credit|loan|indenture)\b",
}


def matches(text: str, patterns: dict[str, str], flags=0) -> list[str]:
    """Noms des motifs présents dans le texte."""
    return [name for name, pattern in patterns.items() if re.search(pattern, text, flags)]


def slices_of(text: str, label_count: int) -> dict[str, str]:
    """Tranches d'une clause. `label_count` : effectif de son label dans le train."""
    n_words = len(text.split())
    return {
        "length": "short" if n_words < 40 else "medium" if n_words < 120 else "long",
        "label_freq": "head" if label_count >= 1000 else "torso" if label_count >= 300 else "tail",
        "jurisdiction": (matches(text, JURISDICTIONS) or ["none"])[0],
        "pronoun": "+".join(matches(text, PRONOUNS, re.IGNORECASE)) or "none",
        "party": "+".join(matches(text, PARTIES)) or "none",
        "family": (matches(text, FAMILIES, re.IGNORECASE) or ["other"])[0],
    }


def compute_slices(texts, labels, train_counts) -> pd.DataFrame:
    """Une ligne par clause, dans l'ordre du split."""
    return pd.DataFrame(slices_of(t, train_counts[y]) for t, y in zip(texts, labels))


def group_ids(names: list[str], groups: list[list[str]]) -> np.ndarray:
    """Identifiant de groupe par label : les quasi-synonymes partagent le même."""
    ids = np.arange(len(names))
    for group in groups:
        members = [names.index(name) for name in group]
        ids[members] = min(members)
    return ids


def label_summary(names: list[str], counts: np.ndarray) -> dict:
    """Déséquilibre des classes : extrêmes et ratio."""
    top, bottom = counts.argmax(), counts.argmin()
    return {
        "n_labels": len(names),
        "most_frequent": {"label": names[top], "count": int(counts[top])},
        "least_frequent": {"label": names[bottom], "count": int(counts[bottom])},
        "imbalance_ratio": round(float(counts[top] / counts[bottom]), 1),
    }


def length_stats(texts) -> dict[str, float]:
    """Longueur des clauses en mots."""
    words = np.array([len(t.split()) for t in texts])
    return {
        "mean": round(float(words.mean()), 1),
        "median": float(np.median(words)),
        "p95": float(np.percentile(words, 95)),
        "max": int(words.max()),
    }
