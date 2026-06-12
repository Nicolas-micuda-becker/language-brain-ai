# Projet : Dimensionnalité Intrinsèque & Morphosyntaxe française

Analyse de la dimensionnalité intrinsèque (ID) des représentations de GPT-2 sur un corpus littéraire français (Gutenberg), en fonction de la composition morphosyntaxique des phrases (VERB, NOUN, ADJ, ADV)

## Structure

```
corpus/raw/          textes bruts Gutenberg (txt)
corpus/processed/    corpus POS (JSON)
scripts/             code
output/              résultats  tests
tests/               plan de tests runner et validation
```

## Le dataset

12 513 phrases françaises de Baudelaire, Verlaine, Maupassant, Molière, La Fontaine.
Chaque phrase a : son texte, sa source, sa longueur (courte/moyenne/longue), ses ratios POS (VERB/NOUN/ADJ/ADV), sa catégorie dominante, ses tokens avec lemme et POS.

Fichier principal : `corpus/processed/corpus_complet.json`

## Pipeline

**1. POS-tagging** 
```bash
python scripts/pos_tagger.py
```
Lit les .txt de corpus/raw/, produit corpus_complet.json

**2. Alignement BPE**
```bash
python scripts/alignement_gpt2.py
python scripts/alignement_gpt2.py --source baudelaire_fleurs_mal --limit 150
```
Aligne les mots spacy avec les sous-mots BPE de GPT-2
Produit : corpus/processed/corpus_alignement_gpt2.json

**3. Extraction des embeddings**
```bash
python scripts/extraction_embeddings_gpt2.py
python scripts/extraction_embeddings_gpt2.py --source moliere_avare --limit 150 --output output/emb_moliere.jsonl
```
Passe les phrases dans GPT-2 récupère les vecteurs
Produit : output/gpt2_embeddings.jsonl

**4. Calcul ID**
```bash
python scripts/calcul_id.py --echelle phrase --dominante VERB
python scripts/calcul_id.py --echelle phrase --longueur courte
python scripts/calcul_id.py --echelle phrase --source baudelaire_fleurs_mal
```
Calcule l'ID (Two-NN + MLE) sur lesvecteurs filtré.
Produit : output/resultats_id.jsonl (1 ligne par run)

## Filtres

```
--source      nom fichier source
--dominante   VERB / NOUN / ADJ / ADV / mixte
--longueur    courte / moyenne / longue
--limit       nombre max de phrases
```

## Décisions techniques

- AUX fusionné dans VERB (auxiliaires = verbes fonctionnels)
- Seuil dominance : 0.25 pour VERB/NOUN, 0.15 pour ADJ/ADV
- Representation d'un mot = moyenne des sous-mots BPE
- Couche GPT-2 = dernière couche caché (hidden states[-1])
- Estimateur minimum : 20 vecteurs requis
