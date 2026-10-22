"""Petits modèles réels (poids aléatoires) construits en local : aucun téléchargement."""

import pandas as pd
import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import modules
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = [
    "the",
    "this",
    "agreement",
    "shall",
    "be",
    "governed",
    "by",
    "laws",
    "of",
    "state",
    "new",
    "york",
    "delaware",
    "either",
    "party",
    "may",
    "terminate",
    "upon",
    "days",
    "written",
    "notice",
    "any",
    "given",
    "in",
    "writing",
    "parties",
    "keep",
    "all",
    "information",
    "confidential",
    "employee",
    "company",
    "his",
    "her",
    "she",
    "he",
    "under",
    "law",
    "is",
    "to",
    "and",
    "a",
    "payment",
    "lease",
    "landlord",
]
N_LABELS = 3
HIDDEN = 16


@pytest.fixture(scope="session")
def tiny_bert(tmp_path_factory):
    """Dossier d'un mini-BERT de classification (3 types) et de son tokenizer."""
    path = tmp_path_factory.mktemp("tiny_bert")
    vocab = {token: i for i, token in enumerate(SPECIALS + WORDS)}
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=N_LABELS,
        id2label={0: "a", 1: "b", 2: "c"},
    )
    BertForSequenceClassification(config).save_pretrained(path)
    BertTokenizer(vocab=vocab).save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_embedder(tiny_bert):
    """Mini modèle d'embeddings (même BERT, pooling moyen), sans téléchargement."""
    return SentenceTransformer(
        modules=[modules.Transformer(str(tiny_bert)), modules.Pooling(HIDDEN)]
    )


def ledgar_from_csv(directory) -> DatasetDict:
    """Mini LEDGAR (train, validation, test) lu depuis des CSV : remplace le téléchargement."""
    features = Features({"text": Value("string"), "label": ClassLabel(names=["a", "b", "c"])})
    return DatasetDict(
        {
            split: Dataset.from_pandas(pd.read_csv(directory / f"{split}.csv"), features=features)
            for split in ("train", "validation", "test")
        }
    )


@pytest.fixture
def load_fixture_ledgar():
    return ledgar_from_csv
