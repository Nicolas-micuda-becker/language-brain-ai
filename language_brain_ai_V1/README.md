# Projet — Dimensionnalité Intrinsèque & Morphosyntaxe française

> Version de travail incomplète — partagée pour montrer les premiers résultats et l'avancement. Le pipeline est fonctionnel mais les tests sont encore en cours.

Analyse de la dimensionnalité intrinsèque (ID) des représentations de GPT-2 sur un corpus littéraire français (Gutenberg), en fonction de la composition morphosyntaxique des phrases (VERB, NOUN, ADJ, ADV).

Les premiers résultats et observations sont dans `RESULTATS.md`.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_md
```

## Structure

```
corpus/raw/          textes bruts Gutenberg (.txt)
corpus/processed/    corpus POS-taggé (JSON)
scripts/             tout le code
output/              résultats des tests (vide — generé par le pipeline)
RESULTATS.md         observations et conclusions des premiers tests
```

## Le dataset

12 513 phrases françaises issues de Baudelaire, Verlaine, Maupassant, Molière, La Fontaine.
Chaque phrase a : son texte, sa source, sa longueur (courte/moyenne/longue), ses ratios POS (VERB/NOUN/ADJ/ADV), sa catégorie dominante, ses tokens avec lemme et POS.

Fichier principal : `corpus/processed/corpus_complet.json`

## Pipeline — dans l'ordre

**1. POS-tagging** (déjà fait, ne pas relancer sauf nouveau corpus)
```bash
python scripts/pos_tagger.py
```

**2. Alignement BPE**
```bash
python scripts/alignement_gpt2.py
python scripts/alignement_gpt2.py --source baudelaire_fleurs_mal --limit 150
```

**3. Extraction des embeddings**
```bash
python scripts/extraction_embeddings_gpt2.py
python scripts/extraction_embeddings_gpt2.py --source moliere_avare --limit 150 --output output/emb_moliere.jsonl
```

**4. Calcul de l'ID**
```bash
python scripts/calcul_id.py --echelle phrase --dominante VERB
python scripts/calcul_id.py --echelle phrase --source baudelaire_fleurs_mal
```

## Décisions techniques

- AUX fusionné dans VERB
- Seuil dominance : 0.25 pour VERB/NOUN, 0.15 pour ADJ/ADV
- Représentation d'un mot = moyenne des sous-mots BPE
- Couche GPT-2 = dernière couche cachée (hidden states[-1])
- Estimateur minimum : 20 vecteurs requis
