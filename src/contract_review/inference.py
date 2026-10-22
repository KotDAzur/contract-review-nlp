"""Inférence : classer des clauses et chercher des clauses similaires, depuis un run entraîné."""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from contract_review.classification import process as classification
from contract_review.retrieval import process as retrieval
from contract_review.utils import read_yaml


class ClausePredictor:
    """Type de clause, confiance, top 3 et `needs_review` (confiance sous le seuil du run)."""

    def __init__(self, run_dir: Path):
        self.cfg = read_yaml(run_dir / "config.yaml")["classification"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = classification.load_classifier(
            run_dir / "models" / "classifier", self.device
        )
        config = self.model.config
        self.names = [config.id2label[i] for i in range(config.num_labels)]
        evaluation = run_dir / "metrics" / "evaluation.json"
        if not evaluation.exists():
            raise SystemExit("Seuil de confiance introuvable : lancez d'abord l'étape evaluation.")
        self.threshold = json.loads(evaluation.read_text())["transformer"]["selective"]["threshold"]

    def predict(self, clauses: list[str]) -> list[dict]:
        loader = classification.make_loader(
            self.tokenizer, clauses, None, self.cfg["max_len"], self.cfg["eval_batch_size"]
        )
        use_amp = self.cfg["fp16"] and self.device.type == "cuda"
        logits = classification.predict_logits(self.model, loader, self.device, use_amp)
        results = []
        for probs in classification.softmax(logits):
            best = np.argsort(-probs)[:3]
            results.append(
                {
                    "type": self.names[best[0]],
                    "confidence": round(float(probs[best[0]]), 4),
                    "top3": [
                        {"type": self.names[i], "confidence": round(float(probs[i]), 4)}
                        for i in best
                    ],
                    "needs_review": bool(probs[best[0]] < self.threshold),
                }
            )
        return results


class ClauseSearcher:
    """Recherche dans la banque de clauses indexée par l'étape retrieval (index déjà construit)."""

    def __init__(self, run_dir: Path, predictor: ClausePredictor):
        cfg = read_yaml(run_dir / "config.yaml")["retrieval"]
        model_name = cfg["hybrid_model"]
        spec = cfg["dense_models"][model_name]
        self.model = SentenceTransformer(spec["name"])
        self.model.max_seq_length = cfg["max_seq_length"]
        self.prefix, self.top_labels = spec["query_prefix"], cfg["top_labels"]
        self.index = faiss.read_index(str(run_dir / "retrieval" / f"{model_name}.faiss"))
        bank = pd.read_parquet(run_dir / "retrieval" / "bank.parquet")
        self.texts, self.labels = bank["text"].tolist(), bank["label"].to_numpy()
        self.predictor = predictor
        self.label_embs = retrieval.encode(self.model, predictor.names)

    def search_question(self, question: str, k: int = 5) -> list[dict]:
        """Question de juriste : routage vers les types proches, contraintes exactes (durée, lieu)."""
        query = retrieval.encode(self.model, [question], self.prefix)[0]
        ids = retrieval.question_search(
            self.index, query, self.label_embs, self.labels, self.texts,
            retrieval.extract_constraints(question), k, self.top_labels,
        )  # fmt: skip
        return self.hits(ids, query)

    def search_clause(self, clause: str, k: int = 5) -> list[dict]:
        """Clause collée : on cherche parmi les clauses du type prédit."""
        query = retrieval.encode(self.model, [clause])[0]
        predicted = self.predictor.names.index(self.predictor.predict([clause])[0]["type"])
        ids = retrieval.search(self.index, query, k, np.flatnonzero(self.labels == predicted))
        return self.hits(ids, query)

    def hits(self, ids: list[int], query: np.ndarray) -> list[dict]:
        return [
            {
                "type": self.predictor.names[self.labels[i]],
                "score": round(float(self.index.reconstruct(int(i)) @ query), 4),
                "text": self.texts[i],
            }
            for i in ids
        ]
