# Plan d'annotation des tests — Étape 5

Ce document liste les tests de l'étape 5 pour explorer l'ID selon la longueur, le POS dominant, la complexité syntaxique et les niveaux d'agrégation.

| ID | Description | Échelle | Filtre POS | Filtre longueur | Complexité | Métriques | Taille échantillon | Notes |
|---:|---|---|---|---|---|---|---:|---|
| T1 | ID vs longueur de phrase | phrase | — | courte/moyenne/longue | — | Two-NN, MLE | >= 100 phrases/bin | courte <5 mots, moyenne 5-12, longue >12 |
| T2 | ID vs POS dominant | phrase | NOUN/VERB/ADJ/mixte | — | — | Two-NN, MLE | >= 100 phrases/catégorie | Dominante = POS le plus fréquent |
| T3 | ID vs complexité syntaxique | phrase | — | — | simple/complexe | Two-NN, MLE | >= 100 phrases/condition | Complexité via propositions ou profondeur |
| T4 | Agrégation mot -> phrase | phrase | — | — | — | moyenne, médiane | toutes | Comparer moyenne vs médiane des ID mots |
| T5 | ID paragraphe vs longueur | paragraphe | — | — | — | Two-NN, MLE | >= 50 par bucket | Bucketiser par nombre de phrases |
| T6 | Comparaison corpus A/B/C | document | — | — | — | ANOVA, t-test | selon corpus | Test de significativité entre genres |
| T7 | Robustesse MLE | mot/phrase | — | — | — | diagnostics MLE | — | Filtrer distances nulles ou quasi nulles |
| T8 | Échelle multi-niveaux | multi | — | — | — | Two-NN, MLE | — | Cohérence mot -> phrase -> paragraphe |

## Priorités
- Haute: T1, T2, T3, T7
- Moyenne: T4, T8
- Basse: T5, T6
