# Data card : LEDGAR (LexGLUE)

Fiche de description du jeu de données, au format « Datasheets for Datasets ». Les chiffres
viennent de l'étape `data` (`artifacts/runs/<date>_<exp>/data/stats.json`), relancée avec
`uv run main.py run --exp legalbert --steps data`.

LEDGAR est un corpus de **paragraphes de contrats américains**, chacun étiqueté par un
seul type de clause (100 types). Il sert ici à entraîner, évaluer et auditer un classifieur de
clauses et un moteur de recherche de clauses.

## Motivation

- **Tâche** : classer une clause de contrat parmi 100 types (governing law, termination,
  confidentiality...), à partir de son seul texte.
- **Pourquoi ce jeu** : c'est le jeu public de référence pour cette tâche, et il fait partie
  du benchmark LexGLUE, ce qui permet de comparer avec des résultats publiés.
- **Ce qu'il ne permet pas** : juger un contrat entier, repérer une clause risquée, ou
  conclure pour d'autres droits que le droit américain (voir « Qui est représenté »).

## Provenance

| | |
|---|---|
| Source des textes | Contrats déposés auprès de la SEC américaine, publics sur EDGAR |
| Corpus d'origine | LEDGAR, Tuggener et al. (2020), LREC |
| Version utilisée | LexGLUE, Chalkidis et al. (2022), config `ledgar` |
| Chargement | `load_dataset("coastalcph/lex_glue", "ledgar")` |
| Version figée | révision `c23fdff1a6bf74e0e1a71cb86f1e781d37da888c` (dans `configs/base.yaml`) |
| Colonnes | `text` (la clause), `label` (un entier de 0 à 99) |
| Langue | anglais |
| Licence | CC BY 4.0, d'après la fiche LexGLUE sur Hugging Face |

Les textes sont récupérés au premier lancement et ne sont **pas redistribués** par ce dépôt.
La révision figée garantit que tous les runs utilisent exactement les mêmes données.

## Composition

| Split | Clauses |
|---|---|
| train | 60 000 |
| validation | 10 000 |
| test | 10 000 |

Une ligne = un paragraphe de contrat = une étiquette. Les titres de section du contrat servent
d'étiquette.

**Construction par LexGLUE.** À partir du corpus LEDGAR d'origine (environ 850 000 clauses),
LexGLUE ne garde que les **100 types les plus fréquents** (environ 80 000 clauses) : les types
rares du corpus d'origine sont donc exclus, et un outil réel en rencontrerait d'autres que le
modèle ne connaît pas. Le découpage est **chronologique** : train 2016-2017, validation 2018,
test 2019. Le test mesure donc un décalage dans le temps, plus exigeant qu'un tirage aléatoire,
mais le vocabulaire et les pratiques de rédaction changent aussi d'une année à l'autre.

### Longueur des clauses (en mots)

| Split | Moyenne | Médiane | 95e centile | Maximum |
|---|---|---|---|---|
| train | 113,9 | 86 | 305 | 1 215 |
| validation | 112,6 | 85 | 296 | 900 |
| test | 107,8 | 78 | 299 | 998 |

Avec le tokenizer de Legal-BERT, la médiane est de 100 tokens et le 95e centile de 349 :
**12,2 % des clauses de validation dépassent les 256 tokens** gardés à l'entraînement et sont
tronquées.

### Déséquilibre des classes

- Plus fréquent : *Governing Laws* (3 167 clauses). Plus rare : *Books* (23 clauses).
  Rapport de 138 contre 1.
- Les 5 plus fréquents : Governing Laws (3 167), Notices (2 493), Counterparts (2 427),
  Entire Agreements (2 340), Severability (1 808).
- Les 5 plus rares : Sanctions (118), Anti-Corruption Laws (106), Qualifications (47),
  Assigns (31), Books (23).

| Fréquence du type dans le train | Types | Clauses du train | Part |
|---|---|---|---|
| head (au moins 1 000) | 16 | 25 248 | 42,1 % |
| torso (300 à 999) | 62 | 30 769 | 51,3 % |
| tail (moins de 300) | 22 | 3 983 | 6,6 % |

Les 22 types rares comptent peu d'exemples : une erreur sur eux pèse peu en accuracy et
beaucoup en macro-F1, qui est la métrique principale du projet.

## Qui est représenté, et qui ne l'est pas

**Représenté** : des sociétés qui déposent leurs contrats auprès de la SEC (essentiellement
des sociétés cotées américaines), en anglais, sous droit américain, dans des contrats
commerciaux, de financement, d'emploi et de fusion-acquisition (d'après nos tranches).
Dans le train, 5 446 clauses citent un État américain, dominées par New York (1 654) et le
Delaware (1 077), loin devant la Californie (522) et le Texas (386).

**Non représenté** :

- les contrats européens et le droit civil (le vocabulaire et la structure des clauses
  diffèrent) ;
- les autres langues, dont le français ;
- les PME et les contrats non déposés auprès de la SEC ;
- les contrats de consommation ;
- les contrats non publics.

Dans le test, seules **27 clauses sur 10 000 mentionnent une juridiction hors États-Unis**
(selon nos regex, qui ne reconnaissent qu'une liste de pays).

## Les étiquettes

- **Elles viennent des titres de section**, pas d'une annotation de leur contenu : une clause
  placée sous le titre « Miscellaneous » ou « General » reçoit ce type quel que soit son
  contenu. Le taux d'erreur de ces étiquettes n'est pas mesuré.
- **Une seule étiquette par clause**, alors qu'une clause peut traiter de plusieurs sujets.
- **Des quasi-synonymes** coexistent, car les contrats nomment leurs sections différemment.
  Les confusions entre eux ne sont pas de vraies erreurs. Ils sont regroupés dans
  `synonym_groups` (`configs/base.yaml`) pour mesurer une accuracy et un retrieval
  *tolérants* :

| Groupe |
|---|
| Indemnifications, Indemnity |
| Waivers, No Waivers |
| Governing Laws, Applicable Laws |
| Amendments, Modifications |
| Entire Agreements, Integration |
| Jurisdictions, Submission To Jurisdiction, Consent To Jurisdiction, Venues |
| Successors, Assigns, Binding Effects, Assignments |
| Tax Withholdings, Withholdings |
| Definitions, Defined Terms |
| Costs, Expenses, Fees |
| Representations, Warranties |

## Métadonnées absentes et tranches déduites

Le jeu ne fournit **aucune** information sur le type de contrat, les parties, la date ou la
société. Pour auditer le modèle, l'étape `data` déduit des *tranches* du texte par regex
(`src/contract_review/data/process.py`). Ce sont des heuristiques : elles peuvent se
tromper, et `metrics/slice_samples.csv` (étape `audit`) permet d'en relire des exemples.

| Tranche | Modalités et effectifs sur le test |
|---|---|
| Longueur | court (moins de 40 mots) 2 206 ; moyen (40 à 119) 4 693 ; long (120 et plus) 3 101 |
| Fréquence du type | head 4 166 ; torso 5 183 ; tail 651 |
| Juridiction | État américain 889 ; hors États-Unis 27 ; aucune 9 084 |
| Pronoms | masculin 299 ; féminin 37 ; les deux 143 ; aucun 9 521 |
| Type de partie | individu 497 ; entreprise 3 195 ; les deux 1 565 ; aucun 4 743 |
| Famille de contrat | crédit 1 911 ; emploi 1 736 ; fusion-acquisition 623 ; bail 328 ; licence 99 ; autre 5 303 |

Les petites tranches (27, 37, 99 clauses) donnent des intervalles de confiance larges : un
écart non significatif sur elles ne prouve pas l'absence de biais. Plus de la moitié des
clauses tombent dans « autre » ou « aucun » : le découpage est grossier.

## Doublons et fuite

- **Aucun doublon exact** : ni dans le train, ni entre le train et la validation ou le test
  (mesuré).
- **449 clauses de test sur 10 000 (4,5 %)** deviennent identiques à une clause du train une
  fois la casse et la ponctuation retirées. L'étape `retrieval` les exclut des résultats, et
  l'étape `audit` donne l'accuracy avec et sans elles.
- **Pré-entraînement** : Legal-BERT a été pré-entraîné, entre autres, sur 76 366 contrats
  américains d'EDGAR, la même source que LEDGAR. Un recouvrement de texte avec le test est
  possible. Il n'est pas mesurable ici, et les étiquettes n'ont pas servi au pré-entraînement.

## Données personnelles

Les contrats sont des documents publics, mais ils peuvent citer des noms de dirigeants, des
adresses ou des rémunérations (contrats d'emploi). Nous n'avons pas vérifié si LEDGAR les a
anonymisés, et nous n'ajoutons aucune anonymisation. Les fichiers qui contiennent des
extraits de clauses (`retrieval/bank.parquet`, `metrics/slice_samples.csv`) restent dans
`artifacts/` (ignoré par git).

## Usages

**Adapté** : recherche et prototypage de classification de clauses et de recherche de clauses
sur des contrats anglais de droit américain ; comparaison avec des résultats LexGLUE.

**À éviter** : tout usage décrit comme interdit dans `MODEL_CARD.md`. En particulier, ne pas
conclure qu'un modèle entraîné ici fonctionne sur des contrats européens, de droit civil ou en
français : ce jeu n'en contient pas.

## Conséquences pour le projet

- Le seuil de confiance est choisi sur la **validation**, jamais sur le test.
- Les résultats sont rapportés aussi **par type** et **par tranche**, pas seulement en
  moyenne, à cause du déséquilibre.
- Les quasi-synonymes justifient une accuracy et un retrieval tolérants, en plus des
  mesures strictes.
- L'absence de contrats européens justifie un audit hors distribution (clauses françaises) et
  la recommandation d'un **pilote annoté** avant tout usage européen.

## Reproduire ces chiffres

```bash
uv run main.py run --exp legalbert --steps data
```

Écrit `data/stats.json`, `data/label_names.json`, `data/label_counts_train.json` et
`data/slices_*.parquet` dans le dossier du run.

## Références

- Tuggener et al. (2020). *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification
  of Legal Provisions in Contracts.* LREC 2020.
- Chalkidis et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in
  English.* ACL 2022.
- Chalkidis et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School.* Findings of
  EMNLP 2020.
