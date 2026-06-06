"""
Calcul de la Dimensionnalité Intrinsèque (ID) sur les embeddings GPT-2.

Deux estimateurs disponibles :
  - two_nn  : Two Nearest Neighbors (Facco et al., 2017)
  - mle     : Maximum Likelihood Estimation (Levina & Bickel, 2004)

Deux échelles :
  - mot     : nuage de tous les vecteurs-mots des phrases sélectionnées
  - phrase  : nuage des vecteurs-phrases (un par phrase)

Usage :
    python scripts/calcul_id.py --echelle mot --estimateur two_nn
    python scripts/calcul_id.py --echelle phrase --estimateur mle --dominante VERB
    python scripts/calcul_id.py --echelle phrase --dominante NOUN --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
DEFAULT_INPUT = BASE_DIR / "output" / "gpt2_embeddings.jsonl"


# ─── Chargement ───────────────────────────────────────────────────────────────

def charger_embeddings(path: Path) -> list[dict]:
    entrees = []
    with path.open(encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                entrees.append(json.loads(ligne))
    return entrees


def filtrer(entrees, source=None, dominante=None, longueur=None, limit=None):
    if source:
        entrees = [e for e in entrees if e.get("source") == source]
    if dominante:
        entrees = [e for e in entrees if e.get("dominante") == dominante]
    if longueur:
        entrees = [e for e in entrees if e.get("longueur") == longueur]
    if limit:
        entrees = entrees[:limit]
    return entrees


def construire_nuage(entrees: list[dict], echelle: str) -> np.ndarray:
    """Construit la matrice de vecteurs selon l'échelle choisie."""
    vecteurs = []
    if echelle == "phrase":
        for e in entrees:
            emb = e.get("phrase_embedding", [])
            if emb:
                vecteurs.append(emb)
    elif echelle == "mot":
        for e in entrees:
            for mot in e.get("words", []):
                emb = mot.get("embedding", [])
                if emb:
                    vecteurs.append(emb)
    else:
        raise ValueError(f"Echelle inconnue : {echelle!r}. Choisir 'mot' ou 'phrase'.")

    if len(vecteurs) < 20:
        raise ValueError(
            f"Seulement {len(vecteurs)} vecteurs — minimum 20 requis pour un estimateur fiable."
        )
    return np.array(vecteurs, dtype=np.float32)


# ─── Estimateurs ──────────────────────────────────────────────────────────────

def two_nn(X: np.ndarray) -> float:
    """
    Estimateur Two Nearest Neighbors (Facco et al., 2017).
    ID = 1 / mean(ln(r2/r1)) où r1, r2 sont les distances aux 2 plus proches voisins.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3).fit(X)
    distances, _ = nn.kneighbors(X)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Évite la division par zéro
    masque = r1 > 0
    mu = r2[masque] / r1[masque]
    mu = mu[mu > 1]  # mu doit être > 1 par définition

    if len(mu) == 0:
        raise ValueError("Impossible de calculer Two-NN : toutes les distances r1 sont nulles.")

    return 1.0 / np.mean(np.log(mu))


def mle(X: np.ndarray, k: int = 5) -> float:
    """
    Estimateur MLE (Levina & Bickel, 2004).
    Pour chaque point, estime l'ID locale via les k plus proches voisins.
    ID = moyenne des ID locales.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)

    id_locales = []
    for i in range(len(X)):
        dk = distances[i, k]
        if dk == 0:
            continue
        dj = distances[i, 1:k]
        dj = dj[dj > 0]
        if len(dj) == 0:
            continue
        id_locale = 1.0 / (np.mean(np.log(dk / dj)))
        id_locales.append(id_locale)

    if not id_locales:
        raise ValueError("Impossible de calculer MLE : distances nulles.")

    return float(np.mean(id_locales))


ESTIMATEURS = {"two_nn": two_nn, "mle": mle}


# ─── Main ─────────────────────────────────────────────────────────────────────

def parser_arguments():
    parser = argparse.ArgumentParser(description="Calcul de l'ID sur les embeddings GPT-2")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--echelle", choices=["mot", "phrase"], default="phrase")
    parser.add_argument("--estimateur", choices=["two_nn", "mle", "les_deux"], default="les_deux")
    parser.add_argument("--source", default=None)
    parser.add_argument("--dominante", default=None)
    parser.add_argument("--longueur", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parser_arguments()

    entrees = charger_embeddings(args.input)
    entrees = filtrer(entrees, args.source, args.dominante, args.longueur, args.limit)
    print(f"Phrases chargées : {len(entrees)}")

    X = construire_nuage(entrees, args.echelle)
    print(f"Nuage : {X.shape[0]} vecteurs de dimension {X.shape[1]}")

    resultats = {}
    estimateurs = ["two_nn", "mle"] if args.estimateur == "les_deux" else [args.estimateur]

    for nom in estimateurs:
        fn = ESTIMATEURS[nom]
        id_val = fn(X)
        resultats[nom] = round(id_val, 4)
        print(f"ID ({nom}) = {id_val:.4f}")

    # Sauvegarde dans output/
    sortie = BASE_DIR / "output" / "resultats_id.jsonl"
    sortie.parent.mkdir(exist_ok=True)
    with sortie.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "echelle": args.echelle,
            "source": args.source,
            "dominante": args.dominante,
            "longueur": args.longueur,
            "n_vecteurs": X.shape[0],
            "dim": X.shape[1],
            **resultats,
        }, ensure_ascii=False) + "\n")
    print(f"Résultat ajouté dans {sortie}")


if __name__ == "__main__":
    main()
