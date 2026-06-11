# Tests — étape 5

Fichiers principaux:
- `tests/plan_tests.md`
- `tests/plan_tests.csv`

Commande utile (WSL):

```bash
cd /chemin/vers/language-brain-ai
python scripts/calcul_id.py --echelle phrase --estimateur les_deux --limit 50
```

Sorties attendues:
- `output/resultats_id.jsonl`

Agrégation des résultats:

```bash
# depuis la racine du dépôt (WSL)
python3 scripts/aggregate_results.py
```

Validation statistique:

```bash
# comparaison A/B/C sur les embeddings de phrase disponibles
python3 scripts/validate_statistical_difference.py --bootstrap 100 --sample-size 150
```
