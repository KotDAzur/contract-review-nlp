# Model card : classifieur de clauses et recherche de clauses

Ce document décrit ce que fait le modèle, comment il a été entraîné et évalué, et surtout
**ce pour quoi il ne doit pas servir**. Les données sont décrites dans `DATA_CARD.md`, les
résultats détaillés et l'audit de biais dans `AUDIT.md` (généré dans le dossier de chaque run).

Les cases marquées *à compléter* se remplissent avec les chiffres du run final : chaque ligne
indique le fichier où les lire.

## Résumé

Le projet classe une clause de contrat parmi **100 types** (governing law, termination,
confidentiality...), indique quand sa réponse doit être **relue par un humain**, et retrouve
des clauses similaires dans une banque de 60 000 clauses, à partir d'une clause collée ou
d'une question de juriste (« termination clauses with a 30-day notice period »).

Il est entraîné et évalué sur LEDGAR : des contrats **américains, en anglais**.

## Composants

| Composant | Modèle | Rôle |
|---|---|---|
| Classifieur principal | `nlpaueb/legal-bert-base-uncased`, affiné (109 M de paramètres) | Prédit le type de la clause |
| Baseline | TF-IDF (1-2 grammes, `sublinear_tf`, `min_df=2`) + SVM linéaire | Ce que le transformer doit battre pour justifier son coût |
| Recherche lexicale | BM25 (`rank_bm25`) | Comparaison |
| Recherche dense | `sentence-transformers/all-MiniLM-L6-v2` et `BAAI/bge-small-en-v1.5`, index FAISS | Comparaison, et base de l'hybride |
| Recherche hybride | BGE + filtre sur le type de clause | Utilisée par la commande `search` |
| Traduction (audit seulement) | `Helsinki-NLP/opus-mt-en-fr` | Produit des clauses françaises de test, ne fait pas partie de l'outil |

**Pourquoi Legal-BERT et pas CamemBERT** : les données sont en anglais, et CamemBERT est un
modèle français. Il ne servirait que si l'on entraînait sur des contrats français, qui
n'existent pas dans ce jeu de données.

Legal-BERT (licence CC BY-SA 4.0) a été pré-entraîné sur 12 Go de textes juridiques anglais :
législation européenne et britannique, jurisprudence, et 76 366 contrats américains d'EDGAR.

## Usage prévu

**Pour qui** : un juriste ou une équipe legal-tech qui revoit des contrats **en anglais, de
droit américain**.

**Pour quoi** :

- pré-étiqueter les clauses d'un contrat avec leur type, en laissant un humain valider ;
- retrouver rapidement des clauses comparables dans une banque de clauses.

```bash
uv run main.py predict "This Agreement shall be governed by the laws of the State of Delaware."
uv run main.py search "termination clauses with a 30-day notice period" --k 5
```

`predict` renvoie le type, la confiance, le top 3 et `needs_review` :

```json
{"type": "...", "confidence": 0.0, "top3": [{"type": "...", "confidence": 0.0}], "needs_review": false}
```

(valeurs fictives, pour montrer le format)

## Ce que l'outil ne doit pas faire

- **Aucune décision juridique automatique.** L'outil propose, un humain décide. Il ne rend
  pas un avis juridique.
- **Pas d'usage sur des contrats non anglophones ou de droit civil** sans réentraînement sur
  des données annotées de ce domaine. L'audit montre ce qui se passe hors distribution
  (`AUDIT.md`, section 4), mais ne remplace pas une validation.
- **Il ne détecte ni les clauses risquées, ni les clauses anormales.** Il dit de quel *type*
  est une clause, pas si elle est bonne, équilibrée ou conforme.
- **Les clauses sous le seuil de confiance partent en revue humaine** (`needs_review`). Ne
  jamais les étiqueter automatiquement.
- **Moins fiable sur les types rares** : 22 des 100 types ont moins de 300 exemples
  d'entraînement (le plus rare en a 23). Surveiller le F1 par type
  (`metrics/per_class.csv`) avant de se fier à un type précis.
- **Pas de conclusion sur les personnes ni sur les sociétés** à partir des tranches de
  l'audit : elles sont déduites du texte par regex et servent uniquement à repérer des écarts
  de performance.

## Entraînement

| | |
|---|---|
| Données | LEDGAR : 60 000 clauses d'entraînement, 10 000 de validation |
| Longueur maximale | 256 tokens (12,2 % des clauses de validation sont tronquées) |
| Optimiseur | AdamW, `lr` 3e-5, `weight_decay` 0,01 (sauf biais et LayerNorm) |
| Planning | échauffement de 10 % puis décroissance linéaire |
| Batch | 32, padding dynamique, `fp16` si CUDA disponible, clipping du gradient à 1,0 |
| Époques | 5 au maximum ; la meilleure époque sur le macro-F1 de validation est conservée. Sur le run final c'est la 5e, et l'arrêt anticipé ne s'est pas déclenché |
| Arrêt anticipé | patience de 2 époques |
| Poids de classes | désactivés par défaut (`class_weights: sqrt_inv` disponible) |
| Seed | 42 (un seul entraînement) |
| Matériel et durée | NVIDIA GeForce RTX 4070, environ 5 minutes par époque (`run_info.json` donne le GPU et la durée de chaque étape) |

La validation sert au choix du meilleur époque et du seuil de confiance. Le jeu de test n'est
utilisé qu'à partir de l'étape `evaluation`, et le seuil n'est jamais choisi dessus.

## Évaluation

Toutes les métriques sont calculées sur les 10 000 clauses de test, avec un intervalle de
confiance à 95 % par bootstrap (1 000 rééchantillons). Le **macro-F1 est la métrique
principale**, parce que les classes sont déséquilibrées.

| | Baseline TF-IDF | Legal-BERT affiné | Écart (avec IC) |
|---|---|---|---|
| Accuracy (= micro-F1) | 0,8809 [0,8742 ; 0,8875] | 0,8777 [0,8713 ; 0,8840] | -0,0032 [-0,0086 ; +0,0015], non significatif |
| Macro-F1 | 0,8284 [0,8133 ; 0,8381] | 0,8021 [0,7894 ; 0,8153] | **-0,0263 [-0,0379 ; -0,0089], significatif** |
| Weighted-F1 | 0,8756 [0,8683 ; 0,8825] | 0,8732 [0,8663 ; 0,8798] | -0,0024 [-0,0082 ; +0,0028], non significatif |
| Accuracy tolérante aux synonymes | 0,9091 | 0,9132 | (pas d'intervalle) |

Source : `metrics/evaluation.json`. L'écart transformer moins baseline est calculé sur les
mêmes rééchantillons, donc apparié : c'est lui qui répond à « le transformer vaut-il son
coût ? ».

**Référence publiée** : pour Legal-BERT sur LEDGAR, LexGLUE rapporte environ **88 de micro-F1
et 83 de macro-F1**. Les protocoles d'entraînement peuvent différer (nombre d'époques,
longueur maximale des séquences, seeds) : la comparaison est indicative.

**Lecture du run final.** Le baseline TF-IDF atteint la référence publiée (88,1 et 82,8).
Legal-BERT affiné pendant 5 époques ne la dépasse pas : accuracy équivalente, et un macro-F1
**plus faible de 2,6 points**, dont l'intervalle de confiance exclut zéro. Sur ce run, le
transformer ne justifie donc pas son coût en qualité de classification. Deux indices
suggèrent un entraînement inachevé plutôt qu'un modèle inadapté : le macro-F1 de validation
progresse encore à la dernière époque (0,709, 0,761, 0,793, 0,810, 0,814) sans que l'arrêt
anticipé se déclenche, et l'écart entre l'accuracy (0,878) et le macro-F1 (0,802) indique que
les types rares sont sous-appris. Ce qui reste en faveur du transformer est sa probabilité
calibrée (ECE de 0,047), qui permet le seuil de revue humaine ci-dessous ; l'équivalent n'a
pas été testé pour le baseline, dont les scores SVM ne sont pas des probabilités.

Les résultats par type (`metrics/per_class.csv`), les 20 confusions les plus fréquentes
(`metrics/confusions.csv`) et les 15 types les plus faibles (`figures/weakest_classes.png`) se
lisent dans le même dossier.

## Confiance et revue humaine

La confiance est la probabilité maximale du modèle. Le seuil est le plus bas qui atteint
**97 % de précision sur la validation** ; l'étape `evaluation` le vérifie ensuite sur le
test. Au-dessus du seuil, la clause peut être étiquetée automatiquement ; en dessous,
`needs_review` vaut `true`.

| | Valeur |
|---|---|
| Seuil de confiance | 0,958 (choisi sur la validation) |
| Part étiquetée automatiquement (test) | 68,4 % (validation : 69,3 %) |
| Précision de ces clauses (test) | 96,7 % (validation : 97,0 %), un peu sous la cible de 97 % |
| ECE (écart moyen entre confiance et exactitude) | 0,047 |

Source : `metrics/evaluation.json` (clés `selective` et `ece`) et `figures/reliability.png`,
`figures/coverage_precision.png`. Si le test ne confirme pas les 97 % obtenus sur la
validation, c'est le chiffre du test qui fait foi.

## Recherche de clauses

Deux situations sont évaluées (`metrics/retrieval.json`), avec P@1, P@5, P@10, MRR@10 et
nDCG@10 :

- **clause vers clauses** : 1 000 clauses de test servent de requêtes ; un résultat est
  pertinent s'il a le **même type** ;
- **question vers clauses** : 25 questions de juristes écrites à la main, avec le type attendu
  et, quand c'est possible, une contrainte exacte (« 30 days », « Delaware »).

Deux précautions de lecture :

- **La mesure est circulaire pour l'hybride clause vers clauses.** Il cherche seulement parmi
  les clauses du type prédit, et « pertinent » veut dire « même type » : son score est en
  pratique borné par l'accuracy du classifieur. Il ne prouve pas à lui seul que les résultats
  sont meilleurs.
- **Les 25 questions** sont écrites et jugées par une seule personne : c'est un contrôle
  qualitatif, pas une statistique.

Les clauses de test identiques au train, à la casse et à la ponctuation près (4,5 %), sont
exclues des résultats.

## Biais et robustesse

`AUDIT.md` (généré par l'étape `audit`) détaille :

- l'accuracy par **tranche** (longueur, fréquence du type, juridiction, pronoms, type de
  partie, famille de contrat), corrigée du mélange de types de clause, avec correction de
  Holm ; les tranches de moins de 50 clauses ne sont pas testées ;
- la **charge de revue humaine** par tranche : quelles clauses partent plus souvent en revue ;
- des **tests contrefactuels** : changer le genre des pronoms, l'État américain par un pays
  européen ou la forme sociale (Inc., LLC par GmbH, S.A.S.), avec un témoin pour distinguer
  l'effet de la perturbation du simple bruit ;
- le **français** : 12 clauses écrites à la main et 300 clauses de test traduites
  automatiquement, avec la part qui serait étiquetée automatiquement et sa précision ;
- la **fuite** entre train et test.

| Constat de l'audit | Résultat |
|---|---|
| Écarts de tranche significatifs | *à compléter* |
| Sensibilité au genre, à la juridiction, à la forme sociale | *à compléter* |
| Anglais contre français : accuracy et part étiquetée automatiquement | *à compléter* |
| Accuracy avec et sans les clauses du test déjà vues | *à compléter* |

## Limites connues

- **Données américaines et anglaises uniquement.** Rien ne garantit que le modèle fonctionne
  sur des contrats européens ; les clauses traduites mesurent l'effet de la langue, pas celui
  du droit civil. Pour des clients européens : **on ne le sait pas, l'audit le montre, et il
  faut un pilote annoté** avant tout usage.
- **Les étiquettes viennent des titres de section** et certains types sont quasi synonymes :
  une partie des « erreurs » n'en sont pas (voir l'accuracy tolérante).
- **Un seul entraînement** (seed 42) : la variabilité entre entraînements n'est pas mesurée.
- **12,2 % des clauses sont tronquées** à 256 tokens.
- **Possible recouvrement avec le pré-entraînement** : Legal-BERT a vu des contrats d'EDGAR,
  la source de LEDGAR. Ce risque n'est pas mesurable ici.
- **Les tranches de l'audit sont des heuristiques** (regex), et les plus petites (27 à 99
  clauses) ont des intervalles larges : un écart non significatif ne prouve pas l'absence de
  biais.
- **Les 25 questions de juristes et les 12 clauses françaises** sont écrites par une seule
  personne.

## Reproduire

```bash
uv sync
uv run main.py run --exp legalbert
```

Chaque run écrit dans `artifacts/runs/<date>_<exp>/` sa configuration complète
(`config.yaml`) et `run_info.json` : seed, hash git, état du dépôt, versions des librairies,
GPU et durée de chaque étape. Le dataset est figé par sa révision, et les tests
(`uv run pytest`) tournent sans réseau.

## Références

- Chalkidis et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School.* Findings of
  EMNLP 2020.
- Chalkidis et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in
  English.* ACL 2022.
- Tuggener et al. (2020). *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification
  of Legal Provisions in Contracts.* LREC 2020.

## Auteurs

Mael Zapata, ThomasFLM
