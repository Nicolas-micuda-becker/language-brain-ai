# Contexte projet — Dimensionnalité Intrinsèque & Morphosyntaxe

## Ce qu'est ce projet

Projet académique (L3) avec Nicolas et Youssef.
Objectif : mesurer la dimensionnalité intrinsèque (ID) des embeddings GPT-2 français en fonction de la composition morphosyntaxique des phrases (VERB, NOUN, ADJ, ADV).
Corpus : 6 textes littéraires français (Gutenberg) — Baudelaire, Verlaine, Maupassant (x2), Molière, La Fontaine.

## Répartition des tâches

**Nicolas** : dataset, POS-tagging, filtrage, analyse des résultats, observations, graphiques, agrégations (moyenne/médiane)
**Youssef** : GPT-2, alignement BPE, extraction embeddings, calcul ID, formalisme mathématique du rapport

## Décisions techniques prises

- Modèle spaCy : `fr_core_news_md`
- AUX fusionné dans VERB
- Seuils dominance : 0.25 pour VERB/NOUN, 0.15 pour ADJ/ADV
- Représentation d'un mot = moyenne des sous-mots BPE
- Couche GPT-2 = dernière couche cachée (hidden states[-1])
- Estimateur minimum : 20 vecteurs
- Modèle GPT-2 : `asi/gpt-fr-cased-small`

## État du projet

- Dataset 12 513 phrases POS-taggé : fait
- Alignement BPE / extraction embeddings : fait (scripts de Youssef)
- Calcul ID (Two-NN + MLE) : fait
- Premiers tests (5 tests) : faits, résultats dans RESULTATS.md
- `language_brain_ai_V1/` : version envoyée à la prof (README + RESULTATS + scripts nettoyés + corpus)

## Fichiers importants

- `corpus/processed/corpus_complet.json` : dataset principal
- `output/gpt2_embeddings.jsonl` : embeddings calculés (900 phrases, 5 sources)
- `output/resultats_id.jsonl` : résultats bruts des runs calcul_id.py
- `RESULTATS.md` : observations et conclusions des tests
- `TESTS.md` : liste complète des tests à faire

## Ce qui reste à faire

- Relancer les tests 2-5 proprement via calcul_id.py (actuellement seulement test 1 dans resultats_id.jsonl)
- Tester sur corpus complet (actuellement 900 phrases sur 12 513)
- Catégorie ADJ pas encore testable (trop peu de phrases dans les échantillons)
- Graphiques et visualisations (côté Nicolas)
- Rapport final
