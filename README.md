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

## Résultats

Chiffres du run final sur le jeu de test (10 000 clauses), avec intervalle de confiance à 95 %.
Le macro-F1 est la métrique principale : les classes sont très déséquilibrées.

| | Baseline TF-IDF | Legal-BERT affiné |
|---|---|---|
| Accuracy | *à compléter* | *à compléter* |
| Macro-F1 | *à compléter* | *à compléter* |
| Part étiquetée automatiquement, à 97 % de précision visée | | *à compléter* |

Source : `metrics/evaluation.json` du run. Le détail (par type de clause, calibration, retrieval,
audit) est dans `MODEL_CARD.md` et `AUDIT.md`.

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
Le dossier du run porte la **date du jour** (`<date>_<exp>`) : relancer une étape le
lendemain crée un nouveau dossier, vide. Pour reprendre un run, le faire le même jour.

La durée de chaque étape est enregistrée dans `run_info.json`.

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

## Documentation

| Document | Contenu |
|---|---|
| [`MODEL_CARD.md`](MODEL_CARD.md) | Les modèles, l'entraînement, l'évaluation, et **ce pour quoi l'outil ne doit pas servir** |
| [`DATA_CARD.md`](DATA_CARD.md) | LEDGAR : provenance, composition, qui est représenté ou non, problèmes de labels |
| `AUDIT.md` | Rapport de biais et de robustesse du run, généré dans son dossier (pas versionné) |

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

Les tests sont hermétiques : ils construisent de mini-modèles à poids aléatoires et de
mini-jeux de données en local, sans réseau. GitHub Actions lance le style et les tests à
chaque pull request et à chaque push sur `master`.

Ils couvrent la logique pure de chaque étape (un test par fonction) et l'exécution réelle de
`data`, `classification` et `evaluation` sur un mini-jeu de données. Les étapes `retrieval`
et `audit` et les commandes `predict` et `search` ne sont pas encore testées de bout en bout.

## Dépannage

- **Le téléchargement depuis Hugging Face échoue** (erreur mentionnant Xet) : définir
  `HF_HUB_DISABLE_XET=1` avant la commande. Sous PowerShell : `$env:HF_HUB_DISABLE_XET=1`.
- **macOS : `No module named 'contract_review'` après `uv sync`** : le fichier qui rend le
  package importable est marqué « caché » et Python l'ignore. Corriger avec
  `chflags -R nohidden .venv`.
- **Sous Windows, vérifier que le GPU est utilisé** :
  `uv run python -c "import torch; print(torch.cuda.is_available())"` doit afficher `True`.
  Sinon, l'entraînement tourne sur CPU et prend des heures.

## Limites

Elles sont détaillées dans `MODEL_CARD.md` et à la fin de chaque `AUDIT.md`. Les principales :

- LEDGAR est anglais et vient de contrats américains (dépôts SEC). Rien ne garantit que le
  modèle fonctionne sur des contrats européens ; les clauses traduites mesurent l'effet de la
  langue, pas celui du droit civil.
- LEDGAR n'a aucune métadonnée sur les parties ou le type de contrat : les tranches sont
  déduites du texte par regex. `metrics/slice_samples.csv` permet d'en relire des exemples.
- Les 25 questions de juristes et les 12 clauses françaises sont écrites et jugées par une
  seule personne : contrôle qualitatif, pas statistique.
- Un seul entraînement (seed 42) : la variabilité entre entraînements n'est pas mesurée.
- Legal-BERT a été pré-entraîné, entre autres, sur des contrats d'EDGAR, la source de LEDGAR :
  un recouvrement de texte avec le jeu de test est possible, mais non mesurable ici.

## Auteurs

Mael Zapata, ThomasFLM
