import json
import random
from pathlib import Path

CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "corpus_complet.json"


def charger():
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def filtrer(dominante=None, longueur=None, source=None, ratio_min=None, n=None, seed=42):
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
    from collections import Counter
    data = charger()
    print(f"Total : {len(data)} phrases\n")
    for cat, n in Counter(p["dominante"] for p in data).most_common():
        print(f"  {cat:8s} : {n}")
    print()
    for src, n in Counter(p["source"] for p in data).most_common():
        print(f"  {src:40s} : {n}")


if __name__ == "__main__":
    stats()
