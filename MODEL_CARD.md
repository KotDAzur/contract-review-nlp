# Model card : classifieur de clauses et recherche de clauses

Ce document décrit ce que fait le modèle, comment il a été entraîné et évalué, et surtout
**ce pour quoi il ne doit pas servir**. Les données sont décrites dans `DATA_CARD.md`, les
résultats détaillés et l'audit de biais dans `AUDIT.md` (généré dans le dossier de chaque run).

Les chiffres viennent de deux runs versionnés dans `results/` : `legalbert` (5 époques, sans
poids de classes) et `legalbert_long` (12 époques, poids de classes `sqrt_inv`). Chaque tableau
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
| Époques | `legalbert` : 5 ; `legalbert_long` : 12. La meilleure époque sur le macro-F1 de validation est conservée (la 5e et la 11e) ; l'arrêt anticipé ne s'est déclenché dans aucun des deux |
| Arrêt anticipé | patience de 2 époques (3 pour `legalbert_long`) |
| Poids de classes | désactivés pour `legalbert`, `sqrt_inv` (racine de l'inverse de la fréquence) pour `legalbert_long` |
| Seed | 42 (un seul entraînement) |
| Matériel et durée | NVIDIA GeForce RTX 4070, environ 5 minutes par époque (27 min pour `legalbert`, 64 min pour `legalbert_long` ; `run_info.json` donne le GPU et la durée de chaque étape) |

La validation sert au choix du meilleur époque et du seuil de confiance. Le jeu de test n'est
utilisé qu'à partir de l'étape `evaluation`, et le seuil n'est jamais choisi dessus.

## Évaluation

Toutes les métriques sont calculées sur les 10 000 clauses de test, avec un intervalle de
confiance à 95 % par bootstrap (1 000 rééchantillons). Le **macro-F1 est la métrique
principale**, parce que les classes sont déséquilibrées.

| | Baseline TF-IDF | Legal-BERT court (5 ép.) | Legal-BERT long (12 ép., poids) |
|---|---|---|---|
| Accuracy (= micro-F1) | 0,8809 [0,8742 ; 0,8875] | 0,8777 [0,8713 ; 0,8840] | 0,8836 [0,8774 ; 0,8897] |
| Macro-F1 | 0,8284 [0,8133 ; 0,8381] | 0,8021 [0,7894 ; 0,8153] | 0,8288 [0,8131 ; 0,8390] |
| Weighted-F1 | 0,8756 [0,8683 ; 0,8825] | 0,8732 [0,8663 ; 0,8798] | 0,8822 [0,8758 ; 0,8885] |
| Accuracy tolérante aux synonymes | 0,9091 | 0,9132 | 0,9153 |
| Écart de macro-F1 avec le baseline | | **-0,0263 [-0,0379 ; -0,0089], significatif** | +0,0004 [-0,0099 ; +0,0114], non significatif |
| Écart d'accuracy avec le baseline | | -0,0032 [-0,0086 ; +0,0015], non significatif | +0,0027 [-0,0023 ; +0,0078], non significatif |

Source : `results/<run>/metrics/evaluation.json`. L'écart transformer moins baseline est calculé sur les
mêmes rééchantillons, donc apparié : c'est lui qui répond à « le transformer vaut-il son
coût ? ».

**Référence publiée** : pour Legal-BERT sur LEDGAR, LexGLUE rapporte environ **88 de micro-F1
et 83 de macro-F1**. Les protocoles d'entraînement peuvent différer (nombre d'époques,
longueur maximale des séquences, seeds) : la comparaison est indicative.

LexGLUE publie aussi un SVM TF-IDF à **87,0 et 81,4** : notre baseline (88,1 et 82,8) le
dépasse, et l'écart publié entre ce SVM et Legal-BERT n'est que de 1,2 et 1,6 point. Le
protocole de LexGLUE est plus long que le nôtre (jusqu'à 20 époques, séquences de 512 tokens,
meilleur de 5 seeds).

**Lecture des deux runs.** Le baseline TF-IDF atteint la référence publiée de Legal-BERT.
Avec 5 époques et sans poids de classes, Legal-BERT ne la dépasse pas : accuracy équivalente
et macro-F1 **plus faible de 2,6 points**, dont l'intervalle de confiance exclut zéro. Le
macro-F1 de validation progressait encore à la dernière époque (0,709, 0,761, 0,793, 0,810,
0,814), ce qui suggérait un entraînement inachevé. Le second run le confirme : avec 12 époques
et des poids de classes, le macro-F1 monte à 0,829 et **rejoint le baseline** (écart de
+0,04 point, non significatif). Il ne le dépasse pas : sur LEDGAR, le transformer n'apporte
pas de gain mesurable en qualité de classification (LexGLUE publie un écart de seulement 1,2
point entre un SVM TF-IDF et Legal-BERT). Un seul seed par run : un écart inférieur à 1 point
est dans le bruit.

**Le prix du second run : la calibration.** L'ECE passe de 0,047 à 0,091 : après 12 époques,
le modèle est sur-confiant (perte d'entraînement de 0,03). Le seuil de confiance monte à 0,998
et la part étiquetée automatiquement baisse de 68,4 % à 65,4 % pour la même précision. Le run
court est donc le mieux calibré, le run long le meilleur en F1 des types rares : aucun ne
gagne sur tout.

**Où se perd l'écart.** Il se concentre sur les types rares. En moyenne par type, le F1 du
transformer est égal à celui du baseline pour les 16 types de tête (0,923 contre 0,921) et
les 62 types intermédiaires (0,847 contre 0,850), mais inférieur de 11,2 points pour les 22
types de moins de 300 exemples (0,587 contre 0,699), ce qui explique à peu près les 2,6
points de macro-F1. Les types les plus faibles sont Applicable Laws (F1 0,105), Assigns et
Books (0, avec 4 et 2 exemples de test seulement). La plupart des erreurs sont des
confusions entre types quasi synonymes : Tax Withholdings et Withholdings (38 et 13 clauses),
Definitions et Defined Terms (25 et 18), Integration vers Entire Agreements (21),
Applicable Laws vers Governing Laws (39).

Avec le run long, le F1 moyen des 22 types rares monte de 0,587 à 0,680 (baseline : 0,699),
celui des types de tête de 0,923 à 0,930, et Books, Assigns et Applicable Laws passent de 0, 0
et 0,11 à 0,67, 0,29 et 0,32.

Les résultats par type (`metrics/per_class.csv`), les 20 confusions les plus fréquentes
(`metrics/confusions.csv`) et les 15 types les plus faibles (`figures/weakest_classes.png`) se
lisent dans le même dossier.

## Confiance et revue humaine

La confiance est la probabilité maximale du modèle. Le seuil est le plus bas qui atteint
**97 % de précision sur la validation** ; l'étape `evaluation` le vérifie ensuite sur le
test. Au-dessus du seuil, la clause peut être étiquetée automatiquement ; en dessous,
`needs_review` vaut `true`.

| | `legalbert` (court) | `legalbert_long` |
|---|---|---|
| Seuil de confiance | 0,958 | 0,998 |
| Part étiquetée automatiquement (test) | 68,4 % (validation : 69,3 %) | 65,4 % (validation : 67,2 %) |
| Précision de ces clauses (test) | 96,7 % (validation : 97,0 %), un peu sous la cible | 97,0 % (validation : 97,0 %) |
| ECE (écart moyen entre confiance et exactitude) | 0,047 | 0,091 |

Source : `metrics/evaluation.json` de chaque run (clés `selective` et `ece`) et `figures/reliability.png`,
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

L'audit a été lancé sur chacun des deux runs ; les deux concordent sur l'essentiel.

| Constat de l'audit | `legalbert` (court) | `legalbert_long` |
|---|---|---|
| Écarts de tranche significatifs (sur 23) | 5 : clauses courtes -2,1 points, parties de type société +2,4, famille crédit +3,3, fusions-acquisitions +4,0, clauses déjà vues dans le train +8,0 | 5 : parties de type société +2,2, particuliers -5,6, crédit +3,1, baux -6,6, clauses déjà vues +9,4 |
| Genre (masculin vers féminin) | 1,0 % de prédictions changées (n = 299) | 2,0 % (n = 299) |
| Juridiction (témoin Texas, puis France, Angleterre, Allemagne) | 0,9 % ; 1,3 à 1,6 % ; seule l'Allemagne est significative (+0,7 point) | 4,5 % ; 5,0 à 5,8 % ; aucun écart au témoin significatif |
| Forme sociale (témoin Ltd., puis GmbH, S.A.S.) | 2,7 % ; 2,7 % ; 5,0 % (+2,3 points, significatif) | 2,3 % ; 1,8 % ; 4,1 % (+1,8 point, non significatif) |
| Anglais contre français : accuracy | 0,883 contre 0,240 (300 clauses traduites automatiquement) | 0,887 contre 0,197 |
| Français : part étiquetée automatiquement | 3,3 %, précision 0,800 | 1,7 %, précision 0,800 |
| Accuracy avec et sans les clauses du test déjà vues (4,5 %) | 0,878 contre 0,873 | 0,884 contre 0,879 |

Lectures :

- **Le français est hors de portée** des deux modèles : l'outil n'est pas utilisable en
  français.
- **Les clauses déjà vues dans le train** sont presque toutes bien classées, ce qui gonfle
  légèrement le score global (0,5 point).
- **Le run long est plus sensible aux perturbations de juridiction** (4,5 % de prédictions
  changées dès le témoin Texas, contre 0,9 %), à cause de la confusion Applicable Laws et
  Governing Laws, que les poids de classes ont rendue plus fréquente. Le surplus par rapport au
  témoin reste, lui, non significatif.
- **Particuliers et baux** deviennent des tranches à écart significatif dans le run long :
  ces écarts d'accuracy sont à surveiller, mais les tranches restent des heuristiques par regex.

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
