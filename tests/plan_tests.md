# Plan de tests — Étape 5

Liste simple des tests prévus pour l'étape 5.

| ID | Test | Échelle | Filtre POS | Filtre longueur | Complexité | Métriques | Taille échantillon | Notes |
|---:|---|---|---|---|---|---|---:|---|
| T1 | ID selon la longueur | phrase | — | courte/moyenne/longue | — | Two-NN, MLE | >= 100 phrases/bin | courte <5 mots, moyenne 5-12, longue >12 |
| T2 | ID selon le POS dominant | phrase | NOUN/VERB/ADJ/mixte | — | — | Two-NN, MLE | >= 100 phrases/catégorie | On garde le POS le plus présent |
| T3 | ID selon la structure | phrase | — | — | simple/complexe | Two-NN, MLE | >= 100 phrases/condition | On regarde surtout les propositions et la profondeur |
| T4 | Passer du mot à la phrase | phrase | — | — | — | moyenne, médiane | toutes | Voir si les deux mesures racontent la même chose |
| T5 | ID des paragraphes | paragraphe | — | — | — | Two-NN, MLE | >= 50 par bucket | Regrouper selon le nombre de phrases |
| T6 | Comparer les corpus | document | — | — | — | ANOVA, t-test | selon corpus | Voir si les genres se séparent bien |
| T7 | Vérifier MLE | mot/phrase | — | — | — | diagnostics MLE | — | Écarter les distances nulles ou trop petites |
| T8 | Comparer les niveaux | multi | — | — | — | Two-NN, MLE | — | Vérifier la cohérence mot -> phrase -> paragraphe |

## Priorités
- Haute: T1, T2, T3, T7
- Moyenne: T4, T8
- Basse: T5, T6
