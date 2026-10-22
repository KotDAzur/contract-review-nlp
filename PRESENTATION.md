# Présentation orale : plan des diapos et texte à dire

Durée : 10 minutes + 5 de questions. Public : le CTO (résultats reproductibles, évaluation
documentée, audit des défaillances, ce que l'outil ne doit pas faire).
Tous les chiffres viennent de `results/` (runs `legalbert` et `legalbert_long`, seed 42, RTX 4070).

Fil rouge à garder en tête : **on ne vend pas un modèle, on rend compte d'une mesure honnête.**
Le résultat n'est pas "BERT gagne" : c'est "voilà ce que ça vaut, où ça casse, et comment on
l'utilise sans risque".

---

## Diapo 1 : Problème et périmètre (0:30)

**Sur la diapo** : titre, une phrase de problème, trois flèches : classer, signaler, retrouver.

**À dire** :
> "On construit le cœur NLP d'un assistant de revue de contrats. Il fait trois choses :
> il dit de quel type est une clause parmi 100, il dit quand il n'est pas sûr et que
> l'humain doit relire, et il retrouve des clauses similaires dans une banque de 60 000.
> Je vais vous montrer ce que ça vaut, où ça casse, et ce qu'il ne faut pas lui demander."

---

## Diapo 2 : Les données (1:00)

**Sur la diapo** : LEDGAR via LexGLUE, 60k / 10k / 10k, figure de répartition des classes,
"137,7 : 1".

**À dire** :
> "Les données sont LEDGAR, des paragraphes de contrats déposés auprès de la SEC américaine,
> 100 types de clauses. Le découpage est chronologique : entraînement 2016-2017, validation
> 2018, test 2019. Donc le test mesure aussi un décalage dans le temps.
> Le point clé, c'est le déséquilibre : Governing Laws a 3 167 exemples, Books en a 23, soit
> un rapport de 138 contre 1. J'ai séparé 16 types de tête, 62 intermédiaires et 22 types
> rares de moins de 300 exemples. Cette séparation revient tout au long de l'exposé.
> Deux précautions : LexGLUE n'a gardé que les 100 types les plus fréquents, donc un outil
> réel verrait des types inconnus ; et 4,5 % des clauses de test ont un équivalent dans le
> train une fois la casse et la ponctuation retirées, que je mesure à part."

---

## Diapo 3 : Méthode et reproductibilité (1:00)

**Sur la diapo** : schéma des 5 étapes (data, classification, evaluation, retrieval, audit),
baseline TF-IDF + SVM, Legal-BERT affiné, "80 tests, CI, config figée, hash git".

**À dire** :
> "Cinq étapes, une commande chacune. Deux modèles : un baseline TF-IDF avec SVM linéaire, et
> Legal-BERT affiné avec une boucle d'entraînement PyTorch écrite à la main : AdamW, échauffement
> puis décroissance linéaire, fp16, clipping, meilleure époque choisie sur le macro-F1 de
> validation. Je compare à un baseline simple parce que la question du CTO est : est-ce que le
> transformer vaut son coût ?
> Côté reproductibilité : seed fixée, révision des données figée, la configuration et le hash
> git sont enregistrés pour chaque étape, 80 tests, intégration continue sur GitHub. Le métrique
> principal est le macro-F1, parce que les classes sont déséquilibrées : l'accuracy seule
> cacherait les types rares."

---

## Diapo 4 : Résultat principal (1:30)

**Sur la diapo** : tableau.

| | Baseline | Legal-BERT 5 ép. | Legal-BERT 12 ép. + poids |
|---|---|---|---|
| Accuracy | 0,881 | 0,878 | 0,884 |
| Macro-F1 | 0,828 | 0,802 (**−2,6 pts, significatif**) | 0,829 (+0,04, non significatif) |
| ECE | | 0,047 | 0,091 |

**À dire** :
> "Voici le résultat, et il n'est pas celui qu'on attendait : le transformer ne bat pas le
> baseline. Premier run, 5 époques : accuracy équivalente, macro-F1 plus faible de 2,6 points,
> et l'intervalle de confiance bootstrap exclut zéro, avec un test apparié sur les mêmes
> rééchantillons. Le second run, 12 époques avec poids de classes, rejoint le baseline mais
> ne le dépasse pas : j'y reviens à la diapo suivante.
> Pour situer : LexGLUE publie 87,0 de micro-F1 pour un SVM TF-IDF et 88,2 pour Legal-BERT,
> soit un écart de seulement 1,2 point. Notre baseline est un peu au-dessus du SVM publié.
> Donc 'BERT n'écrase pas TF-IDF sur LEDGAR' est attendu. Mais 'BERT est moins bon' est un
> signal d'alerte : ça suggère un entraînement incomplet, pas un modèle inadapté.
> Réserve : un seul seed, protocole plus court que LexGLUE, donc comparaison indicative."

---

## Diapo 5 : Pourquoi, et test de l'hypothèse (1:30)

**Sur la diapo** : le tableau par groupe de fréquence.

| Groupe | Types | F1 baseline | F1 Legal-BERT | Écart |
|---|---|---|---|---|
| Queue (<300 ex.) | 22 | 0,699 | 0,587 | **−11,2** |
| Milieu | 62 | 0,850 | 0,847 | −0,3 |
| Tête (≥1000) | 16 | 0,921 | 0,923 | +0,2 |

**À dire** :
> "J'ai cherché où se perd l'écart. Sur 78 types sur 100, les deux modèles sont à égalité.
> Toute la perte vient des 22 types rares : 11 points de F1. 22 types fois 11 points sur 100,
> ça fait à peu près les 2,6 points de macro-F1. Deuxième indice : le macro-F1 de validation
> montait encore à la dernière époque, 0,709, 0,761, 0,793, 0,810, 0,814, sans que l'arrêt
> anticipé se déclenche. L'hypothèse : sous-entraînement des types rares.
> Je l'ai testée avec un second run : 12 époques, patience 3, poids de classes en racine de
> l'inverse de la fréquence. Résultat : le F1 des 22 types rares passe de 0,587 à 0,680, le
> macro-F1 global de 0,802 à 0,829. L'hypothèse est donc confirmée : le premier run était
> sous-entraîné. Mais le transformer rejoint le baseline, il ne le dépasse pas, et l'écart de
> 0,04 point est dans le bruit avec un seul seed.
> Et il y a un prix : la calibration se dégrade, ECE de 0,047 à 0,091. Le modèle devient
> sur-confiant, le seuil de confiance monte à 0,998 et la part de clauses automatisées baisse
> de 68 à 65 %. Aucun des deux modèles ne gagne sur tout : le court est mieux calibré, le long
> meilleur sur les types rares."

---

## Diapo 6 : Où le modèle se trompe (1:00)

**Sur la diapo** : 5 confusions les plus fréquentes, accuracy tolérante 0,913.

**À dire** :
> "Les erreurs ne sont pas aléatoires. La plupart sont des paires de types quasi synonymes :
> Tax Withholdings et Withholdings, 38 et 13 clauses ; Definitions et Defined Terms, 25 et 18 ;
> Integration vers Entire Agreements, 21. Les étiquettes viennent des titres de section, donc
> deux contrats différents peuvent nommer la même clause de deux façons. Un juriste hésiterait
> aussi. Si on tolère les synonymes, l'accuracy passe à 0,913.
> Le cas le plus préoccupant est Applicable Laws, F1 de 0,105 : 39 clauses sont prises pour
> Governing Laws. Et Books et Assigns, F1 de zéro, avec 2 et 4 exemples de test seulement :
> à lire avec prudence, c'est du bruit statistique autant qu'une faiblesse."

---

## Diapo 7 : Confiance et revue humaine (1:00)

**Sur la diapo** : courbe couverture-précision, "68,4 % automatisées, 96,7 % de précision".

**À dire** :
> "(Chiffres du run court ; le run long est dans la MODEL_CARD.) L'outil ne doit pas décider seul. Il donne une probabilité, et je choisis un seuil de
> confiance sur la validation, jamais sur le test : le plus bas qui atteint 97 % de précision.
> Sur le test, 68 % des clauses passent en automatique avec 96,7 % de précision, un peu sous
> la cible de 97 % : c'est le chiffre du test qui fait foi. Les 32 % restantes partent en
> revue humaine. La calibration est correcte, ECE de 0,047.
> C'est le vrai avantage du transformer : le SVM ne donne pas de probabilités, donc le même
> mécanisme n'a pas été testé pour lui."

---

## Diapo 8 : Recherche de clauses (1:00)

**Sur la diapo** : tableau P@10 (strict) clause→clauses : BM25 0,740, MiniLM 0,721, bge 0,759 ;
questions de juristes : bge 0,788, hybride 0,884.

**À dire** :
> "Quatre systèmes : BM25, deux modèles d'embeddings avec index FAISS, et un hybride qui filtre
> d'abord par type de clause. Pour une question de juriste, l'hybride monte à 0,884 de P@10
> contre 0,788 pour bge seul.
> Mais je dois être clair sur une limite : pour la recherche clause vers clauses, l'hybride
> affiche 0,874 partout, de P@1 à P@10. C'est circulaire : il ne cherche que parmi les clauses
> du type prédit, et 'pertinent' veut dire 'même type'. Son score est donc exactement
> l'accuracy du classifieur. Ça ne prouve pas qu'il est meilleur. Et les 25 questions sont
> écrites et jugées par une seule personne : c'est un contrôle qualitatif, pas une statistique."

---

## Diapo 9 : Audit de biais et de robustesse (1:30)

**Sur la diapo** : 4 constats chiffrés.

- 5 écarts significatifs sur 23 tranches (Holm), dont doublons +8,0 pts
- Contrefactuels : 1 % de prédictions changées (genre), 1,3 à 1,6 % (juridiction), S.A.S. 5,0 % contre 2,7 % pour le témoin
- Français : 0,883 → 0,240, seulement 3,3 % étiquetées automatiquement
- Fuite train/test : 4,5 %, accuracy 0,878 avec, 0,873 sans

**À dire** :
> "L'audit a quatre volets. Un : l'accuracy par tranche, corrigée du mélange de types de clause
> et avec correction de Holm. Cinq écarts significatifs sur 23. Le plus gros, +8 points,
> concerne les clauses déjà vues dans le train : le modèle les reconnaît presque toutes, ce
> qui rappelle qu'une partie du score vient de la mémorisation.
> Deux : les tests contrefactuels. Je change le genre des pronoms, l'État américain par un
> pays européen, la forme sociale, et je regarde si la prédiction bouge, avec un témoin pour
> distinguer l'effet réel du simple bruit. Presque rien ne bouge : 1 % pour le genre. Deux
> effets significatifs mais petits : S.A.S. qui change 5 % des prédictions contre 2,7 % pour
> le témoin, et l'Allemagne, +0,7 point.
> Trois, et c'est le plus important pour un client européen : le français. Sur 300 clauses
> traduites automatiquement, l'accuracy tombe de 0,88 à 0,24, et seules 3 % des clauses
> seraient traitées automatiquement. L'outil n'est pas utilisable en français. Attention : la
> traduction automatique mesure la langue, pas le droit civil. Pour affirmer quelque chose sur
> l'Europe, il faut un pilote annoté.
> Quatre : la fuite. 4,5 % de doublons, effet de 0,5 point : faible sur le score global."

---

## Diapo 10 : Ce que l'outil ne doit pas faire (0:30)

**Sur la diapo** : 4 interdits.

**À dire** :
> "Pour conclure, ce que l'outil ne doit pas faire. Aucune décision juridique automatique :
> il propose, un humain décide. Il dit le type d'une clause, pas si elle est risquée ou
> conforme. Pas d'usage sur des contrats non anglophones ou de droit civil sans données
> annotées. Et pas de conclusion sur des personnes ou des sociétés à partir des tranches :
> elles sont déduites du texte par regex, elles servent à repérer des écarts.
> Les clauses sous le seuil partent toujours en revue. Merci, je prends vos questions."

---

## Questions probables du CTO (5 min)

**Pourquoi pas CamemBERT ou un modèle multilingue ?**
> Les données et la tâche sont américaines et en anglais. Un modèle français ne résout pas le
> problème : il faudrait des clauses annotées en droit civil. L'audit français montre l'écart,
> la suite logique est un pilote annoté avec ces clients, pas un changement de modèle à
> l'aveugle.

**Pourquoi garder BERT s'il ne bat pas TF-IDF ?**
> Sur la classification pure, il ne se justifie pas à ce stade. Son intérêt est la probabilité
> calibrée, qui permet le seuil de revue humaine. Le test équivalent pour le baseline n'a pas
> été fait : c'est une limite assumée.

**Les résultats sont-ils reproductibles ?**
> Oui pour le pipeline : données à révision figée, seed, configuration et hash git dans
> `run_info.json`, 80 tests, CI. Non pour la variabilité entre entraînements : un seul seed,
> donc pas d'écart-type. LexGLUE prend le meilleur de 5 seeds ; on ne le fait pas.

**Que se passe-t-il quand le modèle se trompe ?**
> Trois filets : le seuil de confiance (32 % des clauses en revue), la liste des types rares à
> surveiller (`per_class.csv`), et le fait que beaucoup d'erreurs sont des quasi-synonymes sans
> conséquence de fond. Reste un 3,3 % d'erreurs au-dessus du seuil : il faut une revue par
> échantillonnage.

**Pourquoi 97 % de précision comme cible, et pourquoi le test donne 96,7 % ?**
> C'est un choix métier, ajustable. Le seuil est choisi sur la validation pour éviter
> l'optimisme, donc le test peut tomber un peu en dessous : c'est le chiffre du test qui fait
> foi, et c'est précisément ce qu'on voulait montrer.

**Le biais, concrètement ?**
> On n'a pas de métadonnées sur les parties ni sur le type de contrat : les tranches sont des
> regex. Un écart non significatif ne prouve pas l'absence de biais, surtout sur les petites
> tranches (27 à 99 clauses). Ce qu'on peut dire : sur les perturbations testées, le modèle
> bouge très peu.

**Pourquoi l'hybride est-il circulaire ?**
> Voir diapo 8. C'est pour ça que je m'appuie sur les 25 questions pour juger l'hybride, en
> sachant que c'est qualitatif.

**Quelle est la plus grosse limite ?**
> Un seul entraînement, des données américaines, et une évaluation du retrieval faible. Je
> préfère le dire plutôt que laisser le jury la trouver.

---

## À vérifier avant de présenter

- Choisir le modèle montré sur les diapos 7 et 9 (court : mieux calibré ; long : meilleur F1) et le dire à l'oral.
- Relire `results/legalbert/AUDIT.md` et `results/legalbert_long/AUDIT.md` pour les chiffres cités si la config change.
- Répéter à voix haute avec un chronomètre : viser 9 minutes pour garder de la marge.
