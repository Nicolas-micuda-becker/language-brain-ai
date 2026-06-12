python scripts/calcul_id.py --echelle phrase --estimateur les_deux --limit 50

Sorties attendues:
- `output/resultats_id.jsonl`

Agrégation des résultats:
python3 scripts/aggregate_results.py


Validation statistique:
# comparaison A/B/C sur les embeddings de phrase disponibles
python3 scripts/validate_statistical_difference.py --bootstrap 100 --sample-size 150

Visualisations:
python3 scripts/generate_figures.py

