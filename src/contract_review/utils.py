import json
import logging
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parents[2]
LIBRARIES = (
    "torch",
    "transformers",
    "datasets",
    "scikit-learn",
    "sentence-transformers",
    "faiss-cpu",
    "numpy",
)


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(exp: str) -> dict:
    """Config par défaut, surchargée par celle de l'expérience."""
    base = read_yaml(ROOT / "configs" / "base.yaml")
    return deep_merge(base, read_yaml(ROOT / "configs" / "experiments" / f"{exp}.yaml"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Context:
    """Ce que reçoit chaque étape : la config, le dossier du run et un logger."""

    cfg: dict
    run_dir: Path
    log: logging.Logger

    def dir(self, name: str) -> Path:
        path = self.run_dir / name
        path.mkdir(exist_ok=True)
        return path


def make_context(exp: str) -> Context:
    """Crée (ou réutilise) artifacts/runs/<date>_<exp>/ et y copie la config."""
    cfg = load_config(exp)
    run_dir = ROOT / "artifacts" / "runs" / f"{date.today()}_{exp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg, sort_keys=False))

    ctx = Context(cfg, run_dir, logging.getLogger("contract_review"))
    ctx.log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log_file = logging.FileHandler(ctx.dir("logs") / "run.log", encoding="utf-8")
    for handler in (logging.StreamHandler(), log_file):
        handler.setFormatter(formatter)
        ctx.log.addHandler(handler)

    set_seed(cfg["seed"])
    return ctx


def find_run(exp: str) -> Path:
    """Dossier du run le plus récent de l'expérience qui a un classifieur entraîné."""
    runs = sorted((ROOT / "artifacts" / "runs").glob(f"*_{exp}"))
    trained = [run for run in runs if (run / "models" / "classifier").exists()]
    if not trained:
        raise SystemExit(
            f"Aucun run entraîné pour '{exp}' : lancez `uv run main.py run --exp {exp}`."
        )
    return trained[-1]


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], check=False, cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip()


def write_run_info(ctx: Context, durations: dict[str, float]) -> None:
    """Trace de reproductibilité. Les durées s'accumulent si on relance d'autres étapes."""
    path = ctx.run_dir / "run_info.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    info.update(
        seed=ctx.cfg["seed"],
        git_hash=git("rev-parse", "HEAD"),
        git_dirty=bool(git("status", "--porcelain")),
        python=sys.version.split()[0],
        libraries={name: version(name) for name in LIBRARIES},
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    info.setdefault("durations_s", {}).update(durations)
    path.write_text(json.dumps(info, indent=2))
