import re

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from contract_review.data.process import NON_US, US_STATES

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
NUMBER_WORDS |= {10: "ten", 12: "twelve", 15: "fifteen", 30: "thirty", 60: "sixty", 90: "ninety"}
DURATION = re.compile(
    rf"\b(\d+|{'|'.join(NUMBER_WORDS.values())})[- ](?:business |calendar )?(day|week|month|year)s?\b",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_bm25(texts) -> BM25Okapi:
    return BM25Okapi([tokenize(text) for text in texts])


def bm25_search(bm25: BM25Okapi, text: str, k: int) -> np.ndarray:
    """Termes uniques de la requête : deux fois plus rapide, scores quasi identiques."""
    scores = bm25.get_scores(sorted(set(tokenize(text))))
    return np.argsort(-scores)[:k]


def encode(model, texts, prefix: str = "") -> np.ndarray:
    """Embeddings normalisés : le produit scalaire devient la similarité cosinus."""
    return model.encode(
        [prefix + text for text in texts],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(index, query: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[int]:
    """k plus proches voisins, restreints aux ids `allowed` s'il est donné."""
    params = None if allowed is None else faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
    _, ids = index.search(query[None], k, params=params)
    return [i for i in ids[0] if i != -1]


def extract_constraints(question: str) -> list[str]:
    """Regex exactes déduites de la question : durées ("30 days") et juridictions ("Delaware")."""
    constraints = []
    for number, unit in DURATION.findall(question):
        value = {word: n for n, word in NUMBER_WORDS.items()}.get(number.lower()) or int(number)
        forms = [str(value)] + ([NUMBER_WORDS[value]] if value in NUMBER_WORDS else [])
        constraints.append(rf"(?i)\b({'|'.join(forms)})\b.{{0,12}}\b{unit}s?\b")
    places = f"{US_STATES}|{NON_US}".split("|")
    return constraints + [rf"\b{place}\b" for place in places if place in question]


def question_search(index, query, label_embs, bank_labels, bank_texts, constraints, k, top=2):
    """Question -> clauses : routage vers les `top` types les plus proches, puis contraintes."""
    routed = np.argsort(-(label_embs @ query))[:top]
    allowed = np.flatnonzero(np.isin(bank_labels, routed))
    allowed = [i for i in allowed if all(re.search(c, bank_texts[i]) for c in constraints)]
    return search(index, query, k, np.array(allowed, dtype=np.int64))


def rank(search_fn, queries, k: int, copies_of: dict) -> list[list[int]]:
    """Top-k de chaque requête, sans les copies exactes de la requête dans la banque."""
    ranked = []
    for i, text in enumerate(queries):
        copies = set(copies_of.get(text, ()))
        found = [j for j in search_fn(i, k + len(copies)) if j not in copies]
        ranked.append(found[:k])
    return ranked


def relevance(ranked, query_labels, bank_labels, group: np.ndarray, k: int = 10) -> np.ndarray:
    """1 si le résultat a le label de la requête (ou un synonyme, selon `group`), sinon 0."""
    rel = np.zeros((len(ranked), k))
    for i, (found, label) in enumerate(zip(ranked, query_labels)):
        rel[i, : len(found)] = group[bank_labels[np.asarray(found, dtype=int)]] == group[label]
    return rel


def n_relevant(queries, query_labels, bank_labels, copies_of, group) -> np.ndarray:
    """Nombre de résultats pertinents possibles dans la banque (copies exactes exclues)."""
    in_group = np.bincount(group[bank_labels], minlength=len(group))
    return np.array(
        [
            in_group[group[label]]
            - sum(group[bank_labels[j]] == group[label] for j in copies_of.get(text, ()))
            for text, label in zip(queries, query_labels)
        ]
    )


def retrieval_metrics(rel: np.ndarray, n_rel: np.ndarray) -> dict[str, float]:
    """P@k, MRR@10 et nDCG@10 (pertinence binaire)."""
    k = rel.shape[1]
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discounts)[np.clip(n_rel, 1, k) - 1]
    first_hit = np.where(rel.any(axis=1), 1 / (rel.argmax(axis=1) + 1), 0.0)
    return {
        "p@1": rel[:, :1].mean(),
        "p@5": rel[:, :5].mean(),
        "p@10": rel.mean(),
        "mrr@10": first_hit.mean(),
        "ndcg@10": ((rel * discounts).sum(axis=1) / ideal).mean(),
    }


def slice_precision(rel: np.ndarray, slices: pd.DataFrame) -> pd.DataFrame:
    """P@10 selon la tranche de la clause requête."""
    p10 = rel.mean(axis=1)
    rows = [
        {"slice": column, "value": value, "n": len(idx), "p@10": p10[idx].mean()}
        for column in slices
        for value, idx in slices.groupby(column).indices.items()
    ]
    return pd.DataFrame(rows)


def question_scores(ranked, questions, bank_texts, rel: np.ndarray, rel_tolerant: np.ndarray):
    """Par question : P@10 sur le type (strict, tolérant) et part des résultats qui
    contiennent la contrainte annotée (vide si la question n'en a pas)."""
    rows = []
    for found, question, strict, tolerant in zip(ranked, questions, rel, rel_tolerant):
        pattern = question.get("constraint")
        hits = [bool(re.search(pattern, bank_texts[j])) for j in found] if pattern else []
        rows.append(
            {
                "question": question["question"],
                "p@10": strict.mean(),
                "p@10_tolerant": tolerant.mean(),
                "constraint_hit@10": np.sum(hits) / 10 if pattern else np.nan,
            }
        )
    return pd.DataFrame(rows)
