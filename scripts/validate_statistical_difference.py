#!/usr/bin/env python3
"""Validation statistique pour l'étape 7.

Le script compare trois corpus A/B/C définis par défaut comme suit :
- A = poésie (Baudelaire + Verlaine)
- B = théâtre (Molière)
- C = fable (La Fontaine)

Il travaille sur les fichiers `output/emb_*.jsonl` déjà générés et produit :
- `output/statistical_validation.csv`
- `output/statistical_validation.json`

La stratégie utilisée est un bootstrap sur les embeddings de phrase, puis :
- ANOVA à un facteur entre les trois groupes
- t-tests de Welch par paires
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

DEFAULT_GROUPS = {
    "A": ["baudelaire_fleurs_mal", "verlaine_oeuvres"],
    "B": ["moliere_avare"],
    "C": ["lafontaine_fables"],
}


def load_phrase_embeddings(path: Path) -> list[list[float]]:
    vectors: list[list[float]] = []
    if not path.exists():
        return vectors
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            emb = obj.get("phrase_embedding") or obj.get("embedding")
            if emb:
                vectors.append(emb)
    return vectors


def two_nn(x: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3).fit(x)
    distances, _ = nn.kneighbors(x)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    mask = r1 > 0
    mu = r2[mask] / r1[mask]
    mu = mu[mu > 1]
    if len(mu) == 0:
        raise ValueError("Two-NN: distances insuffisantes.")
    return float(1.0 / np.mean(np.log(mu)))


def mle(x: np.ndarray, k: int = 5) -> float:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    distances, _ = nn.kneighbors(x)
    local_ids = []
    for i in range(len(x)):
        dk = distances[i, k]
        if dk <= 0:
            continue
        dj = distances[i, 1:k]
        dj = dj[(dj > 0) & (dj < dk)]
        if len(dj) == 0:
            continue
        local_ids.append(1.0 / np.mean(np.log(dk / dj)))
    if not local_ids:
        raise ValueError("MLE: distances insuffisantes.")
    return float(np.mean(local_ids))


def bootstrap_ids(vectors: list[list[float]], n_bootstrap: int, sample_size: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    x = np.asarray(vectors, dtype=np.float32)
    results = {"two_nn": [], "mle": []}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(x), size=sample_size)
        sample = x[indices]
        results["two_nn"].append(two_nn(sample))
        results["mle"].append(mle(sample))
    return results


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation statistique des corpus A/B/C.")
    parser.add_argument("--bootstrap", type=int, default=100, help="Nombre de rééchantillonnages bootstrap.")
    parser.add_argument("--sample-size", type=int, default=150, help="Taille d'échantillon par bootstrap et par groupe.")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")
    args = parser.parse_args()

    source_cache: dict[str, list[list[float]]] = {}
    for source_name in {source for sources in DEFAULT_GROUPS.values() for source in sources}:
        vectors = load_phrase_embeddings(OUTPUT_DIR / f"emb_{source_name}.jsonl")
        if not vectors:
            raise FileNotFoundError(f"Aucun embedding trouvé pour {source_name!r}.")
        source_cache[source_name] = vectors

    group_vectors: dict[str, list[list[float]]] = {}
    for group_name, sources in DEFAULT_GROUPS.items():
        combined: list[list[float]] = []
        for source_name in sources:
            combined.extend(source_cache[source_name])
        group_vectors[group_name] = combined

    smallest_group = min(len(vectors) for vectors in group_vectors.values())
    sample_size = min(args.sample_size, smallest_group)
    if sample_size < 20:
        raise ValueError(f"Taille d'échantillon trop faible: {sample_size}.")

    bootstraps: dict[str, dict[str, list[float]]] = {}
    for group_name, vectors in group_vectors.items():
        bootstraps[group_name] = bootstrap_ids(vectors, args.bootstrap, sample_size, args.seed)

    rows = []
    summary = {}
    for metric in ("two_nn", "mle"):
        metric_arrays = [bootstraps[group][metric] for group in DEFAULT_GROUPS]
        anova = stats.f_oneway(*metric_arrays)
        metric_summary = {
            "anova_f": float(anova.statistic),
            "anova_p": float(anova.pvalue),
            "groups": {},
            "pairwise": {},
        }

        for group_name in DEFAULT_GROUPS:
            stats_row = summarize(bootstraps[group_name][metric])
            metric_summary["groups"][group_name] = stats_row
            rows.append(
                {
                    "metric": metric,
                    "kind": "group_summary",
                    "group_a": group_name,
                    "group_b": "",
                    "n_bootstrap": args.bootstrap,
                    "sample_size": sample_size,
                    "mean": round(stats_row["mean"], 6),
                    "std": round(stats_row["std"], 6),
                    "p05": round(stats_row["p05"], 6),
                    "p95": round(stats_row["p95"], 6),
                    "anova_f": round(metric_summary["anova_f"], 6),
                    "anova_p": round(metric_summary["anova_p"], 6),
                    "t_stat": "",
                    "t_p": "",
                }
            )

        for group_a, group_b in combinations(DEFAULT_GROUPS.keys(), 2):
            t = stats.ttest_ind(bootstraps[group_a][metric], bootstraps[group_b][metric], equal_var=False)
            metric_summary["pairwise"][f"{group_a}_vs_{group_b}"] = {
                "t": float(t.statistic),
                "p": float(t.pvalue),
            }
            rows.append(
                {
                    "metric": metric,
                    "kind": "pairwise_ttest",
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_bootstrap": args.bootstrap,
                    "sample_size": sample_size,
                    "mean": "",
                    "std": "",
                    "p05": "",
                    "p95": "",
                    "anova_f": round(metric_summary["anova_f"], 6),
                    "anova_p": round(metric_summary["anova_p"], 6),
                    "t_stat": round(float(t.statistic), 6),
                    "t_p": round(float(t.pvalue), 6),
                }
            )

        summary[metric] = metric_summary

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "statistical_validation.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "kind",
                "group_a",
                "group_b",
                "n_bootstrap",
                "sample_size",
                "mean",
                "std",
                "p05",
                "p95",
                "anova_f",
                "anova_p",
                "t_stat",
                "t_p",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_json = OUTPUT_DIR / "statistical_validation.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Groups: {', '.join(DEFAULT_GROUPS.keys())}")
    print(f"Bootstrap: {args.bootstrap} / sample_size: {sample_size}")
    for metric in ("two_nn", "mle"):
        print(f"{metric}: ANOVA p={summary[metric]['anova_p']:.6g}")
        for pair, values in summary[metric]["pairwise"].items():
            print(f"  {pair}: t={values['t']:.4f}, p={values['p']:.6g}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
