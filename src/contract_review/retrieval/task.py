"""Étape retrieval : indexe la banque de clauses (train) et compare les systèmes de recherche."""

import json
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from contract_review.classification.task import predict_probs
from contract_review.data.process import group_ids, normalize
from contract_review.data.task import load_ledgar
from contract_review.retrieval import process
from contract_review.utils import ROOT, read_yaml


@dataclass
class Bank:
    texts: list[str]
    labels: np.ndarray
    copies_of: dict[str, list[int]]  # texte normalisé -> ids des clauses identiques dans la banque


def build_dense(cfg: dict, bank: Bank, names: list[str], out_dir) -> dict[str, dict]:
    """Un index FAISS par modèle d'embeddings, sauvegardé pour la commande `search`."""
    dense = {}
    for name, spec in cfg["dense_models"].items():
        model = SentenceTransformer(spec["name"])
        model.max_seq_length = cfg["max_seq_length"]
        index = process.build_index(process.encode(model, bank.texts))
        faiss.write_index(index, str(out_dir / f"{name}.faiss"))
        dense[name] = {
            "model": model,
            "index": index,
            "prefix": spec["query_prefix"],
            "label_embs": process.encode(model, names),
        }
    return dense


def dense_system(index, embeddings):
    return lambda i, n: process.search(index, embeddings[i], n)


def clause_systems(bank, dense, bm25, texts, predicted, hybrid) -> dict:
    """Clause -> clauses. L'hybride cherche seulement parmi les clauses du type prédit."""
    embeddings = {name: process.encode(d["model"], texts) for name, d in dense.items()}
    label_ids = [np.flatnonzero(bank.labels == label) for label in range(bank.labels.max() + 1)]
    systems = {"bm25": lambda i, n: process.bm25_search(bm25, texts[i], n)}
    systems |= {name: dense_system(d["index"], embeddings[name]) for name, d in dense.items()}
    systems["hybrid"] = lambda i, n: process.search(
        dense[hybrid]["index"], embeddings[hybrid][i], n, label_ids[predicted[i]]
    )
    return systems


def question_systems(bank, dense, bm25, questions, hybrid, top_labels) -> dict:
    """Question -> clauses. L'hybride route vers les types les plus proches et applique
    les contraintes exactes (durée, juridiction) trouvées dans la question."""
    embeddings = {
        name: process.encode(d["model"], questions, d["prefix"]) for name, d in dense.items()
    }
    best = dense[hybrid]
    systems = {"bm25": lambda i, n: process.bm25_search(bm25, questions[i], n)}
    systems |= {name: dense_system(d["index"], embeddings[name]) for name, d in dense.items()}
    systems["hybrid"] = lambda i, n: process.question_search(
        best["index"], embeddings[hybrid][i], best["label_embs"], bank.labels, bank.texts,
        process.extract_constraints(questions[i]), n, top_labels,
    )  # fmt: skip
    return systems


def rounded(scores: dict) -> dict[str, float]:
    return {name: round(float(value), 4) for name, value in scores.items()}


def run(ctx) -> None:
    cfg, k = ctx.cfg["retrieval"], ctx.cfg["retrieval"]["k"]
    ledgar = load_ledgar(ctx.cfg["data"])
    names = ledgar["train"].features["label"].names
    bank = Bank(list(ledgar["train"]["text"]), np.array(ledgar["train"]["label"]), {})
    for i, text in enumerate(bank.texts):
        bank.copies_of.setdefault(normalize(text), []).append(i)
    strict, tolerant = np.arange(len(names)), group_ids(names, ctx.cfg["synonym_groups"])

    # Requêtes : clauses de test tirées au hasard, avec leurs tranches et leur type prédit
    test = ledgar["test"]
    picked = np.random.default_rng(ctx.cfg["seed"]).choice(
        len(test), cfg["n_queries"], replace=False
    )
    texts = list(np.array(test["text"], dtype=object)[picked])
    labels = np.array(test["label"])[picked]
    slices = pd.read_parquet(ctx.dir("data") / "slices_test.parquet").iloc[picked]
    predicted = predict_probs(ctx, texts).argmax(axis=1)

    out_dir = ctx.dir("retrieval")
    bm25 = process.build_bm25(bank.texts)
    dense = build_dense(cfg, bank, names, out_dir)
    pd.DataFrame({"text": bank.texts, "label": bank.labels}).to_parquet(out_dir / "bank.parquet")

    # Clause -> clauses : pertinent = même type (strict) ou même groupe de synonymes (tolérant)
    clauses, slice_tables = {}, []
    needed = {
        mode: process.n_relevant(texts, labels, bank.labels, bank.copies_of, group)
        for mode, group in (("strict", strict), ("tolerant", tolerant))
    }
    for system, search_fn in clause_systems(
        bank, dense, bm25, texts, predicted, cfg["hybrid_model"]
    ).items():
        ranked = process.rank(search_fn, texts, k, bank.copies_of)
        rel = {
            m: process.relevance(ranked, labels, bank.labels, g, k)
            for m, g in (("strict", strict), ("tolerant", tolerant))
        }
        clauses[system] = {m: rounded(process.retrieval_metrics(rel[m], needed[m])) for m in rel}
        by_slice = process.slice_precision(rel["strict"], slices.reset_index(drop=True))
        slice_tables.append(by_slice.assign(system=system))

    # Question -> clauses : les 25 questions de juristes
    questions = read_yaml(ROOT / cfg["queries_file"])
    question_texts = [q["question"] for q in questions]
    question_labels = np.array([names.index(q["label"]) for q in questions])
    per_question, summary = [], {}
    for system, search_fn in question_systems(
        bank, dense, bm25, question_texts, cfg["hybrid_model"], cfg["top_labels"]
    ).items():
        ranked = process.rank(search_fn, question_texts, k, {})
        rel = process.relevance(ranked, question_labels, bank.labels, strict, k)
        rel_tolerant = process.relevance(ranked, question_labels, bank.labels, tolerant, k)
        scores = process.question_scores(ranked, questions, bank.texts, rel, rel_tolerant)
        per_question.append(scores.assign(system=system))
        summary[system] = rounded(
            scores.drop(columns="question").mean()
        )  # NaN ignorés : contrainte facultative

    metrics_dir = ctx.dir("metrics")
    report = {"hybrid_model": cfg["hybrid_model"], "clauses": clauses, "questions": summary}
    (metrics_dir / "retrieval.json").write_text(json.dumps(report, indent=2))
    pd.concat(slice_tables).to_csv(metrics_dir / "retrieval_slices.csv", index=False)
    pd.concat(per_question).to_csv(metrics_dir / "retrieval_questions.csv", index=False)
    ctx.log.info(
        "retrieval, P@10 clause -> clauses : %s",
        {s: v["strict"]["p@10"] for s, v in clauses.items()},
    )
