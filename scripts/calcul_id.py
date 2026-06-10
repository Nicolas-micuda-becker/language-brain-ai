import argparse
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
DEFAULT_INPUT = BASE_DIR / "output" / "gpt2_embeddings.jsonl"


def charger_embeddings(path):
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


def construire_nuage(entrees, echelle):
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
        raise ValueError(f"Echelle inconnue : {echelle!r}")

    if len(vecteurs) < 20:
        raise ValueError(f"Seulement {len(vecteurs)} vecteurs — minimum 20 requis.")
    return np.array(vecteurs, dtype=np.float32)


def two_nn(X):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    distances, _ = nn.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    masque = r1 > 0
    mu = r2[masque] / r1[masque]
    mu = mu[mu > 1]
    if len(mu) == 0:
        raise ValueError("Two-NN : distances r1 nulles.")
    return 1.0 / np.mean(np.log(mu))


def mle(X, k=5):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    id_locales = []
    for i in range(len(X)):
        dk = distances[i, k]
        if dk <= 0:
            continue
        dj = distances[i, 1:k]
        dj = dj[(dj > 0) & (dj < dk)]
        if len(dj) == 0:
            continue
        id_locales.append(1.0 / np.mean(np.log(dk / dj)))
    if not id_locales:
        raise ValueError("MLE : distances nulles.")
    return float(np.mean(id_locales))


ESTIMATEURS = {"two_nn": two_nn, "mle": mle}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--echelle", choices=["mot", "phrase"], default="phrase")
    parser.add_argument("--estimateur", choices=["two_nn", "mle", "les_deux"], default="les_deux")
    parser.add_argument("--source", default=None)
    parser.add_argument("--dominante", default=None)
    parser.add_argument("--longueur", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    entrees = charger_embeddings(args.input)
    entrees = filtrer(entrees, args.source, args.dominante, args.longueur, args.limit)
    print(f"Phrases : {len(entrees)}")

    X = construire_nuage(entrees, args.echelle)
    print(f"Nuage : {X.shape[0]} vecteurs x {X.shape[1]} dim")

    resultats = {}
    noms = ["two_nn", "mle"] if args.estimateur == "les_deux" else [args.estimateur]
    for nom in noms:
        val = ESTIMATEURS[nom](X)
        resultats[nom] = round(val, 4)
        print(f"ID ({nom}) = {val:.4f}")

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
    print(f"Sauvegarde : {sortie}")


if __name__ == "__main__":
    main()
