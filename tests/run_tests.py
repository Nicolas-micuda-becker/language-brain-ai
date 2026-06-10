#!/usr/bin/env python3
"""Exécution simple des tests T1/T2/T7 pour l'étape 5."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/yjaafri/language-brain-ai-venv/bin/python")
CALCUL_ID = BASE / "scripts" / "calcul_id.py"
OUTPUT = BASE / "output" / "resultats_id.jsonl"
RESULTS_DIR = BASE / "tests" / "results"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open(encoding="utf-8"))


def load_new_rows(before: int) -> list[dict]:
    rows: list[dict] = []
    if not OUTPUT.exists():
        return rows
    with OUTPUT.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index >= before:
                rows.append(json.loads(line))
    return rows


def run_calc(args: list[str]) -> None:
    subprocess.run([str(PYTHON), str(CALCUL_ID), *args], cwd=str(BASE), check=True)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    before = count_lines(OUTPUT)
    for longueur in ["courte", "moyenne", "longue"]:
        run_calc(["--echelle", "phrase", "--estimateur", "les_deux", "--longueur", longueur, "--limit", "50"])
    rows = load_new_rows(before)
    write_csv(rows, RESULTS_DIR / "agg_T1_longueur.csv")

    before = count_lines(OUTPUT)
    for dominante in ["NOUN", "VERB", "ADJ", "mixte"]:
        run_calc(["--echelle", "phrase", "--estimateur", "les_deux", "--dominante", dominante, "--limit", "50"])
    rows = load_new_rows(before)
    write_csv(rows, RESULTS_DIR / "agg_T2_dominante.csv")

    before = count_lines(OUTPUT)
    run_calc(["--echelle", "mot", "--estimateur", "mle", "--limit", "1000"])
    rows = load_new_rows(before)
    write_csv(rows, RESULTS_DIR / "agg_T7_mle_mot.csv")


if __name__ == "__main__":
    main()
