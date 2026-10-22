import json
import logging
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from contract_review import utils

TEST_DATA = Path(__file__).parent / "test_data"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Faux dépôt : configs/base.yaml et configs/experiments/exp.yaml, ROOT pointe dessus."""
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    shutil.copy(TEST_DATA / "base.yaml", tmp_path / "configs" / "base.yaml")
    shutil.copy(TEST_DATA / "exp.yaml", tmp_path / "configs" / "experiments" / "exp.yaml")
    monkeypatch.setattr(utils, "ROOT", tmp_path)
    return tmp_path


def test_read_yaml():
    assert utils.read_yaml(TEST_DATA / "base.yaml")["seed"] == 7


def test_deep_merge():
    base = {"a": {"x": 1, "y": 2}, "b": 3}

    merged = utils.deep_merge(base, {"a": {"y": 20}, "c": 4})

    assert merged == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}


def test_load_config(project):
    cfg = utils.load_config("exp")

    assert cfg["seed"] == 7
    assert cfg["data"]["subset"] == "ledgar"
    assert cfg["classification"] == {"epochs": 1, "model": "tiny"}


def test_set_seed():
    utils.set_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    utils.set_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_make_context(project):
    ctx = utils.make_context("exp")

    assert ctx.run_dir.parent == project / "artifacts" / "runs"
    assert ctx.run_dir.name.endswith("_exp")
    assert yaml.safe_load((ctx.run_dir / "config.yaml").read_text())["seed"] == 7
    assert ctx.dir("data").is_dir()
    for handler in list(ctx.log.handlers):  # libère le fichier de log
        handler.close()
        ctx.log.removeHandler(handler)


def test_find_run(project):
    runs = project / "artifacts" / "runs"
    (runs / "2026-10-20_exp" / "models" / "classifier").mkdir(parents=True)
    (runs / "2026-10-22_exp" / "models" / "classifier").mkdir(parents=True)
    (runs / "2026-10-23_exp").mkdir(parents=True)  # le plus récent, mais sans classifieur

    assert utils.find_run("exp").name == "2026-10-22_exp"


def test_git(tmp_path, monkeypatch):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(utils, "ROOT", tmp_path)

    assert utils.git("rev-parse", "--is-inside-work-tree") == "true"


def test_write_run_info(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "ROOT", tmp_path)
    ctx = utils.Context(cfg={"seed": 42}, run_dir=tmp_path, log=logging.getLogger("test"))

    utils.write_run_info(ctx, {"data": 1.5})
    # Même run repris sur une autre machine : l'étape suivante a son propre GPU
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch.cuda, "get_device_name", lambda index: "GPU de test")
    utils.write_run_info(ctx, {"classification": 60.0})

    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["seed"] == 42
    assert info["steps"]["data"]["duration_s"] == 1.5
    assert info["steps"]["data"]["python"] == sys.version.split()[0]
    assert "torch" in info["steps"]["data"]["libraries"]
    assert info["steps"]["data"]["gpu"] != "GPU de test"  # l'étape déjà écrite garde sa trace
    assert info["steps"]["classification"]["gpu"] == "GPU de test"
