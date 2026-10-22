# Audit de biais et de robustesse

## Constats

- Tranches : 5 écart(s) significatif(s) sur 23 tranches testables (correction de Holm, seuil 0.05, corrigé du mélange de types de clause).
  - length = short : -2.1 points par rapport à l'attendu (n = 2206)
  - party = entity : +2.4 points par rapport à l'attendu (n = 3195)
  - family = credit : +3.3 points par rapport à l'attendu (n = 1911)
  - family = m_and_a : +4.0 points par rapport à l'attendu (n = 623)
  - duplicate_in_train = yes : +8.0 points par rapport à l'attendu (n = 449)
- gender: masculine -> feminine : 1.0 % de prédictions changées (n = 299).
- gender: feminine -> masculine : 0.0 % de prédictions changées (n = 37).
- gender: feminine -> masculine, sans 'her' : 0.0 % de prédictions changées (n = 8).
- jurisdiction: Texas (témoin) : 0.9 % de prédictions changées (n = 867).
- jurisdiction: England and Wales : 1.4 % de prédictions changées (n = 867), écart au témoin +0.5 points (pas de différence significative).
- jurisdiction: France : 1.3 % de prédictions changées (n = 867), écart au témoin +0.3 points (pas de différence significative).
- Governing Laws: US state -> France : 0.4 % de prédictions changées (n = 497).
- jurisdiction: Germany : 1.6 % de prédictions changées (n = 867), écart au témoin +0.7 points (plus que le témoin, significatif).
- company form: Ltd. (témoin) : 2.7 % de prédictions changées (n = 219).
- company form: GmbH : 2.7 % de prédictions changées (n = 219), écart au témoin +0.0 points (pas de différence significative).
- company form: S.A.S. : 5.0 % de prédictions changées (n = 219), écart au témoin +2.3 points (plus que le témoin, significatif).
- Français (test traduit, n = 300) : accuracy 0.883 en anglais contre 0.240 en français ; 3.3 % des clauses françaises seraient étiquetées automatiquement, avec une précision de 0.800.
- Fuite : 4.5 % des clauses de test sont identiques à une clause du train (casse et ponctuation exclues) ; accuracy 0.878 avec elles, 0.873 sans.

## 1. Accuracy par tranche (transformer)

`expected` est l'accuracy attendue vu les types de clause de la tranche (accuracy de chaque type sur l'ensemble du test) ; `gap` = accuracy - expected, avec son intervalle de confiance bootstrap.

| slice | value | n | accuracy | expected | gap | ci_low | ci_high | p_adj | status |
|---|---|---|---|---|---|---|---|---|---|
| length | long | 3101 | 0.880 | 0.867 | 0.013 | 0.002 | 0.023 | 0.264 | ok |
| length | medium | 4693 | 0.883 | 0.881 | 0.002 | -0.007 | 0.010 | 1.000 | ok |
| length | short | 2206 | 0.864 | 0.886 | -0.021 | -0.033 | -0.009 | 0.009 | écart |
| label_freq | head | 4166 | 0.951 | 0.951 | 0.000 | -0.006 | 0.006 | 1.000 | ok |
| label_freq | tail | 651 | 0.647 | 0.647 | -0.000 | -0.033 | 0.034 | 1.000 | ok |
| label_freq | torso | 5183 | 0.848 | 0.848 | 0.000 | -0.009 | 0.009 | 1.000 | ok |
| jurisdiction | non_us | 27 | 0.926 | 0.895 | 0.031 | -0.048 | 0.087 | - | n insuffisant |
| jurisdiction | none | 9084 | 0.880 | 0.879 | 0.000 | -0.006 | 0.006 | 1.000 | ok |
| jurisdiction | us_state | 889 | 0.856 | 0.862 | -0.006 | -0.022 | 0.010 | 1.000 | ok |
| pronoun | feminine | 37 | 0.811 | 0.854 | -0.043 | -0.167 | 0.067 | - | n insuffisant |
| pronoun | masculine | 299 | 0.849 | 0.866 | -0.017 | -0.057 | 0.020 | 1.000 | ok |
| pronoun | masculine+feminine | 143 | 0.790 | 0.834 | -0.043 | -0.107 | 0.015 | 1.000 | ok |
| pronoun | none | 9521 | 0.880 | 0.879 | 0.001 | -0.004 | 0.007 | 1.000 | ok |
| party | entity | 3195 | 0.894 | 0.870 | 0.024 | 0.013 | 0.034 | 0.009 | écart |
| party | individual | 497 | 0.811 | 0.857 | -0.047 | -0.078 | -0.016 | 0.050 | ok |
| party | individual+entity | 1565 | 0.866 | 0.870 | -0.004 | -0.020 | 0.012 | 1.000 | ok |
| party | none | 4743 | 0.878 | 0.887 | -0.010 | -0.018 | -0.002 | 0.211 | ok |
| family | credit | 1911 | 0.897 | 0.864 | 0.033 | 0.021 | 0.046 | 0.009 | écart |
| family | employment | 1736 | 0.858 | 0.876 | -0.017 | -0.033 | -0.003 | 0.264 | ok |
| family | lease | 328 | 0.790 | 0.843 | -0.054 | -0.096 | -0.012 | 0.197 | ok |
| family | license | 99 | 0.848 | 0.893 | -0.044 | -0.114 | 0.018 | 1.000 | ok |
| family | m_and_a | 623 | 0.912 | 0.872 | 0.040 | 0.019 | 0.061 | 0.015 | écart |
| family | other | 5303 | 0.879 | 0.886 | -0.007 | -0.014 | 0.001 | 1.000 | ok |
| duplicate_in_train | no | 9551 | 0.873 | 0.877 | -0.004 | -0.010 | 0.002 | 1.000 | ok |
| duplicate_in_train | yes | 449 | 0.973 | 0.893 | 0.080 | 0.063 | 0.096 | 0.009 | écart |

Écarts significatifs du baseline TF-IDF : 

| slice | value | n | accuracy | expected | gap | ci_low | ci_high | p_adj | status |
|---|---|---|---|---|---|---|---|---|---|
| party | entity | 3195 | 0.897 | 0.874 | 0.022 | 0.012 | 0.032 | 0.009 | écart |
| party | none | 4743 | 0.878 | 0.891 | -0.014 | -0.022 | -0.006 | 0.030 | écart |
| family | credit | 1911 | 0.907 | 0.871 | 0.036 | 0.024 | 0.048 | 0.009 | écart |
| family | lease | 328 | 0.771 | 0.840 | -0.069 | -0.110 | -0.028 | 0.016 | écart |
| duplicate_in_train | yes | 449 | 0.987 | 0.895 | 0.091 | 0.078 | 0.104 | 0.009 | écart |

## 2. Charge de revue humaine par tranche (transformer)

Seuil de confiance de production : 0.958. `auto_rate` = part étiquetée automatiquement ; le reste part en revue humaine.

| slice | value | n | auto_rate | auto_precision |
|---|---|---|---|---|
| length | long | 3101 | 0.713 | 0.967 |
| length | medium | 4693 | 0.701 | 0.966 |
| length | short | 2206 | 0.605 | 0.969 |
| label_freq | head | 4166 | 0.789 | 0.988 |
| label_freq | tail | 651 | 0.339 | 0.783 |
| label_freq | torso | 5183 | 0.642 | 0.958 |
| jurisdiction | non_us | 27 | 0.593 | 1.000 |
| jurisdiction | none | 9084 | 0.720 | 0.967 |
| jurisdiction | us_state | 889 | 0.316 | 0.968 |
| pronoun | feminine | 37 | 0.676 | 0.920 |
| pronoun | masculine | 299 | 0.642 | 0.953 |
| pronoun | masculine+feminine | 143 | 0.629 | 0.911 |
| pronoun | none | 9521 | 0.686 | 0.968 |
| party | entity | 3195 | 0.746 | 0.972 |
| party | individual | 497 | 0.608 | 0.957 |
| party | individual+entity | 1565 | 0.672 | 0.967 |
| party | none | 4743 | 0.653 | 0.964 |
| family | credit | 1911 | 0.742 | 0.975 |
| family | employment | 1736 | 0.667 | 0.962 |
| family | lease | 328 | 0.576 | 0.931 |
| family | license | 99 | 0.727 | 0.944 |
| family | m_and_a | 623 | 0.726 | 0.982 |
| family | other | 5303 | 0.669 | 0.966 |
| duplicate_in_train | no | 9551 | 0.679 | 0.966 |
| duplicate_in_train | yes | 449 | 0.784 | 0.989 |

## 3. Tests contrefactuels

Seules les clauses modifiées par la perturbation sont comptées. `flip_excess` est l'écart de changements par rapport au témoin, sur les mêmes clauses.

| perturbation | model | n | flip_rate | flip_rate_low | flip_rate_high | flip_excess | flip_excess_low | flip_excess_high | accuracy_change | confidence_change | top_flips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gender: masculine -> feminine | baseline | 299 | 0.013 | 0.003 | 0.027 | - | - | - | -0.010 | - | Representations -> Indemnifications (1); Binding Effects -> Successors (1); Duties -> Positions (1) |
| gender: masculine -> feminine | transformer | 299 | 0.010 | 0.000 | 0.023 | - | - | - | -0.007 | 0.003 | Employment -> Positions (1); Terms -> Positions (1); General -> Death (1) |
| gender: feminine -> masculine | baseline | 37 | 0.027 | 0.000 | 0.081 | - | - | - | 0.027 | - | Releases -> Representations (1) |
| gender: feminine -> masculine | transformer | 37 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.004 |  |
| gender: feminine -> masculine, sans 'her' | baseline | 8 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | - |  |
| gender: feminine -> masculine, sans 'her' | transformer | 8 | 0.000 | 0.000 | 0.000 | - | - | - | 0.000 | 0.000 |  |
| jurisdiction: Texas (témoin) | baseline | 867 | 0.024 | 0.015 | 0.035 | - | - | - | 0.000 | - | Governing Laws -> Venues (3); Organizations -> Existence (2); Organizations -> Authority (1) |
| jurisdiction: Texas (témoin) | transformer | 867 | 0.009 | 0.003 | 0.016 | - | - | - | 0.000 | -0.006 | Submission To Jurisdiction -> Jurisdictions (3); Governing Laws -> Jurisdictions (1); Submission To Jurisdiction -> Consent To Jurisdiction (1) |
| jurisdiction: England and Wales | baseline | 867 | 0.025 | 0.015 | 0.037 | 0.001 | -0.010 | 0.013 | -0.003 | - | Venues -> Governing Laws (7); Applicable Laws -> Governing Laws (1); Consent To Jurisdiction -> Venues (1) |
| jurisdiction: England and Wales | transformer | 867 | 0.014 | 0.007 | 0.022 | 0.005 | -0.002 | 0.012 | -0.001 | 0.002 | Submission To Jurisdiction -> Jurisdictions (2); Governing Laws -> Jurisdictions (1); Construction -> Governing Laws (1) |
| jurisdiction: France | baseline | 867 | 0.039 | 0.027 | 0.052 | 0.015 | 0.002 | 0.028 | -0.008 | - | Governing Laws -> Applicable Laws (15); Governing Laws -> Severability (3); Governing Laws -> Waiver Of Jury Trials (2) |
| Governing Laws: US state -> France | baseline | 497 | 0.036 | 0.020 | 0.054 | - | - | - | -0.022 | - | Governing Laws -> Applicable Laws (13); Applicable Laws -> Waiver Of Jury Trials (1); Governing Laws -> Binding Effects (1) |
| jurisdiction: France | transformer | 867 | 0.013 | 0.006 | 0.021 | 0.003 | -0.002 | 0.009 | -0.003 | -0.006 | Governing Laws -> Jurisdictions (2); Submission To Jurisdiction -> Jurisdictions (2); Construction -> Governing Laws (1) |
| Governing Laws: US state -> France | transformer | 497 | 0.004 | 0.000 | 0.010 | - | - | - | -0.002 | -0.010 | Governing Laws -> Jurisdictions (1); Interpretations -> Construction (1) |
| jurisdiction: Germany | baseline | 867 | 0.020 | 0.010 | 0.029 | -0.005 | -0.014 | 0.005 | 0.003 | - | Governing Laws -> Severability (2); Governing Laws -> Jurisdictions (2); Applicable Laws -> Waiver Of Jury Trials (1) |
| jurisdiction: Germany | transformer | 867 | 0.016 | 0.008 | 0.025 | 0.007 | 0.001 | 0.014 | -0.001 | -0.003 | Submission To Jurisdiction -> Jurisdictions (3); Governing Laws -> Jurisdictions (2); Construction -> Governing Laws (1) |
| company form: Ltd. (témoin) | baseline | 219 | 0.027 | 0.009 | 0.050 | - | - | - | 0.009 | - | Amendments -> Subsidiaries (2); Submission To Jurisdiction -> Consent To Jurisdiction (1); Interpretations -> Authority (1) |
| company form: Ltd. (témoin) | transformer | 219 | 0.027 | 0.009 | 0.050 | - | - | - | -0.009 | 0.002 | Anti-Corruption Laws -> Books (1); Successors -> Binding Effects (1); Employment -> Duties (1) |
| company form: GmbH | baseline | 219 | 0.023 | 0.005 | 0.046 | -0.005 | -0.014 | 0.000 | 0.005 | - | Amendments -> Subsidiaries (2); Submission To Jurisdiction -> Consent To Jurisdiction (1); Interpretations -> Authority (1) |
| company form: GmbH | transformer | 219 | 0.027 | 0.009 | 0.050 | 0.000 | 0.000 | 0.000 | -0.009 | 0.002 | Anti-Corruption Laws -> Financial Statements (1); Successors -> Binding Effects (1); Employment -> Duties (1) |
| company form: S.A.S. | baseline | 219 | 0.023 | 0.005 | 0.046 | -0.005 | -0.014 | 0.000 | 0.005 | - | Amendments -> Subsidiaries (2); Submission To Jurisdiction -> Consent To Jurisdiction (1); Interpretations -> Authority (1) |
| company form: S.A.S. | transformer | 219 | 0.050 | 0.023 | 0.078 | 0.023 | 0.005 | 0.041 | 0.000 | -0.006 | Anti-Corruption Laws -> Books (1); Tax Withholdings -> Withholdings (1); Successors -> Binding Effects (1) |

## 4. Hors distribution : français

Les lignes `manuel` portent sur 12 clauses écrites à la main (qualitatif). Les lignes `test traduit` comparent les mêmes clauses de test en anglais et en français automatique.

| model | set | n | accuracy | accuracy_tolerant | mean_confidence | auto_rate | auto_precision | accuracy_change | accuracy_change_low | accuracy_change_high |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | manuel EN | 12.000 | 1.000 | 1.000 | 0.859 | 0.583 | 1.000 | - | - | - |
| baseline | manuel FR | 12.000 | 0.000 | 0.083 | 0.536 | 0.083 | 0.000 | - | - | - |
| baseline | test traduit, original EN | 300.000 | 0.877 | 0.900 | 0.935 | 0.660 | 0.955 | - | - | - |
| baseline | test traduit, FR | 300.000 | 0.110 | 0.127 | 0.403 | 0.033 | 0.300 | - | - | - |
| baseline | test traduit, FR moins EN | - | - | - | - | - | - | -0.767 | -0.813 | -0.717 |
| transformer | manuel EN | 12.000 | 1.000 | 1.000 | 0.859 | 0.583 | 1.000 | - | - | - |
| transformer | manuel FR | 12.000 | 0.417 | 0.583 | 0.536 | 0.083 | 1.000 | - | - | - |
| transformer | test traduit, original EN | 300.000 | 0.883 | 0.917 | 0.935 | 0.660 | 0.955 | - | - | - |
| transformer | test traduit, FR | 300.000 | 0.240 | 0.273 | 0.403 | 0.033 | 0.800 | - | - | - |
| transformer | test traduit, FR moins EN | - | - | - | - | - | - | -0.643 | -0.697 | -0.587 |

Détail des 12 clauses écrites à la main :

| lang | expected | transformer | confidence | baseline | text |
|---|---|---|---|---|---|
| fr | Governing Laws | Interpretations | 0.746 | Expenses | Le présent contrat est régi par le droit français et interprété conformément à celui-ci. |
| fr | Confidentiality | Confidentiality | 0.258 | Counterparts | Chaque partie s'engage à conserver confidentielles les informations confidentielles de l'a |
| fr | Terminations | Terms | 0.228 | Modifications | Chaque partie peut résilier le présent contrat à tout moment moyennant un préavis écrit de |
| fr | Notices | Notices | 0.989 | Modifications | Toute notification prévue par le présent contrat doit être faite par écrit et envoyée par  |
| fr | Severability | Severability | 0.457 | Terms | Si une stipulation du présent contrat est déclarée nulle ou inapplicable, les autres stipu |
| fr | Entire Agreements | Miscellaneous | 0.637 | Modifications | Le présent contrat constitue l'intégralité de l'accord entre les parties sur son objet et  |
| fr | Assignments | Terminations | 0.259 | Counterparts | Aucune partie ne peut céder ou transférer ses droits ou obligations au titre du présent co |
| fr | Counterparts | Counterparts | 0.668 | Amendments | Le présent contrat peut être signé en plusieurs exemplaires, chacun étant considéré comme  |
| fr | Amendments | Modifications | 0.893 | Modifications | Toute modification du présent contrat doit faire l'objet d'un avenant écrit signé par les  |
| fr | Waivers | Effective Dates | 0.168 | Severability | Le fait pour une partie de ne pas exercer un droit prévu au présent contrat ne constitue p |
| fr | Indemnifications | Indemnity | 0.794 | Litigations | Le prestataire s'engage à indemniser et à garantir le client contre toute réclamation, per |
| fr | Jurisdictions | Jurisdictions | 0.335 | Modifications | Tout litige relatif au présent contrat relève de la compétence exclusive des tribunaux de  |
| en | Governing Laws | Governing Laws | 0.927 | Governing Laws | This Agreement shall be governed by and construed in accordance with the laws of France. |
| en | Confidentiality | Confidentiality | 0.997 | Confidentiality | Each party shall keep confidential all confidential information of the other party and sha |
| en | Terminations | Terminations | 0.987 | Terminations | Either party may terminate this Agreement at any time upon thirty (30) days' prior written |
| en | Notices | Notices | 0.999 | Notices | Any notice required or permitted under this Agreement shall be in writing and sent by cert |
| en | Severability | Severability | 0.994 | Severability | If any provision of this Agreement is held to be invalid or unenforceable, the remaining p |
| en | Entire Agreements | Entire Agreements | 0.619 | Entire Agreements | This Agreement constitutes the entire agreement between the parties with respect to its su |
| en | Assignments | Assignments | 0.994 | Assignments | Neither party may assign or transfer any of its rights or obligations under this Agreement |
| en | Counterparts | Counterparts | 0.999 | Counterparts | This Agreement may be executed in several counterparts, each of which shall be deemed an o |
| en | Amendments | Amendments | 0.981 | Amendments | No amendment of this Agreement shall be effective unless made in writing and signed by bot |
| en | Waivers | Waivers | 0.779 | Waivers | The failure of a party to exercise any right under this Agreement shall not constitute a w |
| en | Indemnifications | Indemnifications | 0.541 | Indemnifications | The Contractor shall indemnify and hold harmless the Client from and against any claim, lo |
| en | Jurisdictions | Jurisdictions | 0.494 | Jurisdictions | Any dispute arising out of this Agreement shall be subject to the exclusive jurisdiction o |

## 5. Fuite entre train et test

| model | duplicate_share | accuracy_all | accuracy_without_duplicates | accuracy_without_duplicates_low | accuracy_without_duplicates_high |
|---|---|---|---|---|---|
| baseline | 0.045 | 0.881 | 0.876 | 0.869 | 0.882 |
| transformer | 0.045 | 0.878 | 0.873 | 0.867 | 0.880 |

## 6. Retrieval : P@10 selon la tranche de la clause requête

| slice | value | n | bge | bm25 | hybrid | minilm |
|---|---|---|---|---|---|---|
| family | credit | 174 | 0.781 | 0.791 | 0.908 | 0.744 |
| family | employment | 178 | 0.713 | 0.688 | 0.865 | 0.671 |
| family | lease | 33 | 0.627 | 0.588 | 0.788 | 0.573 |
| family | license | 13 | 0.862 | 0.908 | 1.000 | 0.869 |
| family | m_and_a | 61 | 0.793 | 0.780 | 0.902 | 0.746 |
| family | other | 541 | 0.769 | 0.741 | 0.865 | 0.733 |
| jurisdiction | non_us | 3 | 0.867 | 1.000 | 1.000 | 0.733 |
| jurisdiction | none | 912 | 0.758 | 0.735 | 0.875 | 0.720 |
| jurisdiction | us_state | 85 | 0.771 | 0.782 | 0.859 | 0.736 |
| label_freq | head | 408 | 0.883 | 0.854 | 0.944 | 0.854 |
| label_freq | tail | 68 | 0.491 | 0.466 | 0.691 | 0.485 |
| label_freq | torso | 524 | 0.697 | 0.686 | 0.844 | 0.649 |
| length | long | 305 | 0.754 | 0.767 | 0.889 | 0.718 |
| length | medium | 469 | 0.774 | 0.743 | 0.878 | 0.735 |
| length | short | 226 | 0.735 | 0.695 | 0.845 | 0.698 |
| party | entity | 303 | 0.778 | 0.778 | 0.911 | 0.733 |
| party | individual | 53 | 0.609 | 0.579 | 0.755 | 0.566 |
| party | individual+entity | 167 | 0.728 | 0.693 | 0.874 | 0.696 |
| party | none | 477 | 0.775 | 0.750 | 0.864 | 0.740 |
| pronoun | feminine | 4 | 0.875 | 0.850 | 1.000 | 0.875 |
| pronoun | masculine | 33 | 0.691 | 0.664 | 0.909 | 0.673 |
| pronoun | masculine+feminine | 16 | 0.631 | 0.525 | 0.812 | 0.575 |
| pronoun | none | 947 | 0.763 | 0.746 | 0.873 | 0.725 |

## Limites

- Les tranches sont déduites du texte par regex : ce sont des heuristiques, pas des attributs
  réels (voir `slice_samples.csv` pour relire des exemples). LEDGAR n'a aucune métadonnée sur
  les parties ni sur le type de contrat.
- Le test de genre ne remplace que des pronoms. Il ne dit rien de biais plus subtils, et "her"
  (him ou his) est résolu par une heuristique.
- Un écart non significatif ne prouve pas l'absence de biais : les petites tranches ont des
  intervalles larges, et celles de moins de n clauses ne sont pas testées.
- Les clauses françaises traduites automatiquement mesurent la langue, pas le droit civil
  européen, dont le vocabulaire et la structure diffèrent. Cet audit ne permet pas d'affirmer
  que l'outil fonctionne pour des clients européens : seul un pilote annoté le montrerait.
- Un seul entraînement (seed 42) : la variabilité entre entraînements n'est pas mesurée.
