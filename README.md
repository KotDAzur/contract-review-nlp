# contract-review-nlp

Assistant de revue de contrats entraîné sur LEDGAR : il classe une clause parmi 100 types,
dit quand sa réponse doit être relue par un humain, retrouve des clauses similaires à partir
d'une clause ou d'une question de juriste, et audite ses propres biais.

Projet de groupe du cours de NLP (M1).

## Ce que fait le projet

Le pipeline enchaîne cinq étapes. Chacune lit les sorties des précédentes dans le dossier du run.

| Étape | Ce qu'elle fait | Sorties principales |
|---|---|---|
| `data` | Charge LEDGAR (LexGLUE, révision figée), calcule statistiques et *tranches* de chaque clause (longueur, fréquence du label, juridiction, pronoms, type de partie, famille de contrat) | `data/stats.json`, `data/slices_*.parquet` |
| `classification` | Baseline TF-IDF + SVM linéaire, puis fine-tuning de Legal-BERT (early stopping sur le macro-F1 de validation) | `models/baseline.joblib`, `models/classifier/` |
| `evaluation` | Accuracy et macro-F1 sur le test avec IC bootstrap, écart transformer − baseline, calibration (ECE), seuil de confiance visant 97 % de précision, résultats par classe et par tranche | `metrics/evaluation.json`, `figures/*.png` |
| `retrieval` | Compare BM25, deux modèles denses (MiniLM, BGE) et un système hybride, pour clause → clauses (1 000 requêtes) et question → clauses (25 questions de juristes) | `metrics/retrieval.json`, index FAISS |
| `audit` | Écarts d'accuracy par tranche corrigés du mélange de types (Holm), contrefactuels (genre, juridiction, forme sociale, avec témoins), clauses françaises (12 écrites à la main, 300 traduites automatiquement), fuite train/test | `AUDIT.md`, `metrics/audit_*.csv` |

Le jeu de test n'est utilisé qu'à partir de l'étape `evaluation`. Le seuil de confiance est
choisi sur la validation, puis vérifié sur le test.

Deux choix reviennent dans toutes les étapes :

- **Labels quasi synonymes.** Les titres de section de LEDGAR varient d'un contrat à l'autre
  (`Indemnifications` / `Indemnity`, `Governing Laws` / `Applicable Laws`...). Les groupes de
  `synonym_groups` dans `configs/base.yaml` servent à l'accuracy et au retrieval *tolérants*.
- **Doublons normalisés.** LEDGAR n'a aucun doublon exact, mais 4,5 % des clauses de test
  ont un équivalent dans le train une fois la casse et la ponctuation retirées. Le retrieval
  exclut ces copies des résultats ; l'audit mesure l'accuracy avec et sans elles.

## Installation

Prérequis : Python 3.11 et [uv](https://docs.astral.sh/uv/). Un GPU NVIDIA est fortement
conseillé pour le fine-tuning ; sans GPU, tout tourne sur CPU, mais l'entraînement est long.

```bash
git clone https://github.com/KotDAzur/contract-review-nlp
cd contract-review-nlp
uv sync
```

Sous Windows, `uv sync` installe la build CUDA 12.6 de PyTorch (celle de PyPI est CPU
seulement). Ailleurs, c'est la build PyPI standard.

Le dataset et les modèles (Legal-BERT, MiniLM, BGE, MarianMT) sont téléchargés depuis le Hub
Hugging Face au premier lancement.

## Utilisation

### Lancer le pipeline

```bash
uv run main.py run --exp legalbert                       # les cinq étapes
uv run main.py run --exp legalbert --steps evaluation audit   # seulement certaines
```

`--exp` désigne un fichier de `configs/experiments/`. Les étapes demandées s'exécutent
toujours dans l'ordre `data → classification → evaluation → retrieval → audit`. Relancer une
étape seule suppose que les précédentes ont déjà produit leurs fichiers dans le même run.

### Classer une clause

```bash
uv run main.py predict "This Agreement shall be governed by the laws of the State of Delaware."
```

Renvoie le type prédit, sa confiance, le top 3, et `needs_review: true` quand la confiance
est sous le seuil choisi par l'étape `evaluation`.

### Chercher des clauses

```bash
uv run main.py search "termination clauses with a 30-day notice period" --k 5
uv run main.py search "Either party may terminate this Agreement..." --clause
```

Sans `--clause`, le texte est une question : elle est routée vers les types de clause les plus
proches, puis filtrée par les contraintes exactes qu'elle contient (durée, État). Avec
`--clause`, la recherche se fait parmi les clauses du type prédit.

`predict` et `search` utilisent le run entraîné le plus récent de `--exp` (par défaut
`legalbert`). `search` a besoin que l'étape `retrieval` ait tourné.

## Où trouver les résultats

Chaque run écrit dans `artifacts/runs/<date>_<exp>/` (ignoré par git) :

```
config.yaml        config complète utilisée
run_info.json      seed, hash git, versions des librairies, GPU, durée de chaque étape
AUDIT.md           rapport d'audit : constats chiffrés, tableaux, limites
logs/run.log
data/  models/  metrics/  figures/  retrieval/
```

Pour lire les résultats, commencer par `metrics/evaluation.json`, `metrics/retrieval.json`
puis `AUDIT.md`.

## Configuration

`configs/base.yaml` contient tous les paramètres par défaut (seed, hyperparamètres,
modèles de retrieval, paramètres de l'audit). Un fichier de `configs/experiments/` ne
contient que ce qu'il change ; il est fusionné par-dessus la base.

| Fichier | Rôle |
|---|---|
| `configs/base.yaml` | Paramètres par défaut |
| `configs/experiments/legalbert.yaml` | Expérience principale : `nlpaueb/legal-bert-base-uncased` |
| `configs/experiments/smoke_test.yaml` | Vide pour l'instant, donc identique à la base |
| `configs/lawyer_queries.yaml` | 25 questions de juristes annotées (type attendu, contrainte regex) |
| `configs/ood_clauses.yaml` | 12 clauses françaises et leur équivalent anglais |

Pour une nouvelle expérience, créer `configs/experiments/<nom>.yaml` avec les seules clés à
changer, puis `uv run main.py run --exp <nom>`.

## Structure du code

```
main.py                     CLI : run, predict, search
src/contract_review/
  utils.py                  config, contexte du run, seed, traces de reproductibilité
  inference.py              ClausePredictor et ClauseSearcher (commandes predict/search)
  data/ classification/ evaluation/ retrieval/ audit/
    task.py                 l'étape : lit les entrées, appelle process, écrit les sorties
    process.py              fonctions pures, testées unitairement
tests/                      même arborescence que src/
```

Chaque `task.py` expose une fonction `run(ctx)` ; `ctx` donne la config, le dossier du run et
un logger.

## Tests

```bash
uv run pytest
uv run ruff check src main.py tests
uv run ruff format --check src main.py tests
```

Les 81 tests sont hermétiques : ils construisent de mini-modèles à poids aléatoires et de
mini-jeux de données en local, sans réseau. GitHub Actions lance le style et les tests à
chaque pull request et à chaque push sur `master`.

## Limites

Elles sont détaillées à la fin de chaque `AUDIT.md`. Les principales :

- LEDGAR est anglais et vient de contrats américains (dépôts SEC). Rien ne garantit que le
  modèle fonctionne sur des contrats européens ; les clauses traduites mesurent l'effet de la
  langue, pas celui du droit civil.
- LEDGAR n'a aucune métadonnée sur les parties ou le type de contrat : les tranches sont
  déduites du texte par regex. `metrics/slice_samples.csv` permet d'en relire des exemples.
- Les 25 questions de juristes et les 12 clauses françaises sont écrites et jugées par une
  seule personne : contrôle qualitatif, pas statistique.
- Un seul entraînement (seed 42) : la variabilité entre entraînements n'est pas mesurée.

## Auteurs

Mael Zapata, ThomasFLM
