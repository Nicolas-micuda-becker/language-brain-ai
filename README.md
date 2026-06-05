# Projet ID — Dimensionnalité Intrinsèque & Morphosyntaxe

## Structure du projet

```
projet_ID/
├── corpus/
│   ├── raw/                  # Textes bruts Gutenberg (.txt)
│   └── processed/
│       ├── corpus_complet.json        # TOUT le dataset — point d'entrée principal
│       ├── baudelaire_fleurs_mal_tagged.json
│       ├── maupassant_horla_tagged.json
│       ├── maupassant_boule_de_suif_tagged.json
│       ├── moliere_avare_tagged.json
│       ├── verlaine_oeuvres_tagged.json
│       └── lafontaine_fables_tagged.json
├── scripts/
│   ├── pos_tagger.py         # Pipeline spaCy — génère le dataset (Nicolas)
│   └── filtrer_corpus.py     # Utilitaire de filtrage — point d'entrée pour Youssef
├── output/                   # Tes graphiques et résultats vont ici (Youssef)
├── TESTS.md                  # Liste des observations à produire
└── README.md                 # Ce fichier
```

---

## Le dataset — `corpus_complet.json`

12 513 phrases issues de 6 textes littéraires français (Gutenberg).
Chaque entrée est une phrase avec :

```json
{
  "source": "baudelaire_fleurs_mal",
  "phrase": "Homme libre, toujours tu chériras la mer!",
  "n_mots": 7,
  "longueur": "courte",
  "ratios_pos": {
    "VERB": 0.1429,
    "NOUN": 0.2857,
    "ADJ": 0.1429,
    "ADV": 0.0
  },
  "dominante": "NOUN",
  "tokens": [
    {"texte": "Homme", "lemme": "homme", "pos": "NOUN"},
    {"texte": "libre", "lemme": "libre", "pos": "ADJ"},
    ...
  ]
}
```

### Champs utiles

| Champ | Description |
|---|---|
| `source` | Fichier d'origine (= auteur/texte) |
| `longueur` | `courte` (≤8 mots) / `moyenne` (9-18) / `longue` (>18) |
| `ratios_pos` | Ratio continu 0→1 pour VERB, NOUN, ADJ, ADV |
| `dominante` | Catégorie majoritaire (ou `mixte`) |
| `tokens` | Détail token par token avec lemme et POS tag spaCy |

### Distribution

| Dominante | Phrases |
|---|---|
| mixte | 5076 |
| NOUN | 3869 |
| VERB | 2632 |
| ADV | 576 |
| ADJ | 360 |

---

## Répartition des tâches

### Youssef — GPT-2, Embeddings & ID

1. Charger les phrases via `scripts/filtrer_corpus.py`
2. Importer GPT-2 français et extraire les embeddings (dernière couche cachée)
3. Représenter chaque mot par la moyenne de ses sous-mots BPE
4. Implémenter l'estimateur d'ID (Two-NN ou MLE)
5. Calculer l'ID pour chaque sous-corpus retourné par `filtrer_corpus.py`
6. Sauvegarder les résultats bruts (valeurs d'ID) dans `output/`
7. Rédaction du rapport

---

### Nicolas — Outputs, Agrégation & Analyse

- Comparaisons moyenne / médiane des ID par groupe
- Graphiques et visualisations
- Observations et conclusions pour chaque test (voir `TESTS.md`)

---

## Notes techniques

- **Modèle GPT-2 recommandé** : `antoiloui/gpt2-french` ou `asi/gpt-fr-cased-small` sur HuggingFace
- **Représentation d'un mot** : moyenne des embeddings des sous-mots BPE qui le composent
- **ID minimale** : il faut au moins ~20 vecteurs pour un estimateur fiable — ne pas calculer sur moins d'une dizaine de tokens
- **Couche** : dernière couche cachée de GPT-2 (hidden states)
