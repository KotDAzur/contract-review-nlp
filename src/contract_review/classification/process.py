import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

NO_DECAY = ("bias", "LayerNorm.weight")


class Examples(Dataset):
    """Clauses déjà tokenisées : un dict (input_ids, attention_mask, labels) par clause."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def make_baseline():
    """TF-IDF + SVM linéaire : ce que le transformer doit battre pour se justifier."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    return make_pipeline(tfidf, LinearSVC())


def scores(labels, preds) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def class_weights(counts: np.ndarray, mode: str | None) -> torch.Tensor | None:
    """`sqrt_inv` : poids en 1/racine(effectif), normalisés autour de 1."""
    if mode != "sqrt_inv":
        return None
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(weights / weights.mean(), dtype=torch.float)


def make_loader(tokenizer, texts, labels, max_len, batch_size, shuffle=False, seed=0) -> DataLoader:
    """Tokenise sans padding ; le collator complète chaque batch à sa plus longue clause.
    `labels=None` pour prédire sur des clauses sans étiquette."""
    encoded = tokenizer(list(texts), truncation=True, max_length=max_len)
    examples = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
    ]
    if labels is not None:
        for example, label in zip(examples, labels):
            example["labels"] = label
    return DataLoader(
        Examples(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def make_optimizer(model, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Pas de weight decay sur les biais ni sur les LayerNorm."""
    params = list(model.named_parameters())
    return torch.optim.AdamW(
        [
            {
                "params": [p for n, p in params if not n.endswith(NO_DECAY)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in params if n.endswith(NO_DECAY)], "weight_decay": 0.0},
        ],
        lr=lr,
    )


@torch.no_grad()
def predict_logits(model, loader, device, use_amp=False) -> np.ndarray:
    model.eval()
    logits = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits.append(model(**inputs).logits.float().cpu())
    return torch.cat(logits).numpy()


def train_model(model, train_loader, val_loader, val_labels, cfg, weights, device, log):
    """Boucle d'entraînement. Renvoie les poids du meilleur epoch (macro-F1 validation)."""
    use_amp = cfg["fp16"] and device.type == "cuda"
    model.to(device)
    optimizer = make_optimizer(model, cfg["lr"], cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))

    best_f1, best_state, history, stale = -1.0, None, [], 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(**batch).logits.float(), labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        preds = predict_logits(model, val_loader, device, use_amp).argmax(axis=1)
        epoch_scores = {"epoch": epoch, "train_loss": train_loss / len(train_loader)}
        epoch_scores |= {f"val_{k}": v for k, v in scores(val_labels, preds).items()}
        history.append(epoch_scores)
        log.info("%s", epoch_scores)

        if epoch_scores["val_macro_f1"] > best_f1:
            best_f1, stale = epoch_scores["val_macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                log.info("early stopping")
                break
    return best_state, history
