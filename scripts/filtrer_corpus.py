"""
Usage :
    from scripts.filtrer_corpus import filtrer

    phrases = filtrer(dominante="VERB")
    phrases = filtrer(dominante="NOUN", longueur="courte")
    phrases = filtrer(source="baudelaire_fleurs_mal")
    phrases = filtrer(ratio_min={"VERB": 0.35})
    phrases = filtrer(dominante="VERB", n=100)
"""

import json
import random
from pathlib import Path

CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "corpus_complet.json"

def charger():
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

def filtrer(
    dominante: str = None,
    longueur: str = None,
    source: str = None,
    ratio_min: dict = None,
    n: int = None,
    seed: int = 42,
) -> list[str]:
    """
    Filtre le corpus et retourne une liste de phrases (texte brut).

    Paramètres :
        dominante  : "VERB" | "NOUN" | "ADJ" | "ADV" | "mixte"
        longueur   : "courte" | "moyenne" | "longue"
        source     : nom du fichier source (sans extension)
        ratio_min  : ex. {"VERB": 0.30} — garde les phrases où ratio_VERB >= 0.30
        n          : nombre max de phrases à retourner (tirage aléatoire si excédent)
        seed       : graine pour la reproductibilité du tirage

    Retourne :
        Liste de strings (textes bruts des phrases filtrées)
    """
    data = charger()

    if dominante:
        data = [p for p in data if p["dominante"] == dominante]
    if longueur:
        data = [p for p in data if p["longueur"] == longueur]
    if source:
        data = [p for p in data if p["source"] == source]
    if ratio_min:
        for pos, seuil in ratio_min.items():
            data = [p for p in data if p["ratios_pos"].get(pos, 0) >= seuil]

    if n and len(data) > n:
        random.seed(seed)
        data = random.sample(data, n)

    return [p["phrase"] for p in data]


def stats():
    """Affiche un résumé du corpus pour vérification rapide."""
    data = charger()
    from collections import Counter
    print(f"Total : {len(data)} phrases\n")
    print("Par dominante :")
    for cat, n in Counter(p["dominante"] for p in data).most_common():
        print(f"  {cat:8s} : {n}")
    print("\nPar longueur :")
    for lon, n in Counter(p["longueur"] for p in data).most_common():
        print(f"  {lon:8s} : {n}")
    print("\nPar source :")
    for src, n in Counter(p["source"] for p in data).most_common():
        print(f"  {src:40s} : {n}")


if __name__ == "__main__":
    stats()
    print("\n--- Exemple : 5 phrases VERB courtes ---")
    for p in filtrer(dominante="VERB", longueur="courte", n=5):
        print(" ", p)
