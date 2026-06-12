#!/usr/bin/env python3
"""T1/T2/T7"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import argparse


BASE = Path(__file__).resolve().parent.parent
PYTHON = Path(os.environ.get("PYTHON", sys.executable))
ALIGN_ID = BASE / "scripts" / "alignement_gpt2.py"
EXTRACT_ID = BASE / "scripts" / "extraction_embeddings_gpt2.py"
CALCUL_ID = BASE / "scripts" / "calcul_id.py"
OUTPUT = BASE / "output" / "resultats_id.jsonl"
RESULTS_DIR = BASE / "tests" / "results"
TMP_ALIGN = RESULTS_DIR / "tmp_align.json"
TMP_EMBED = RESULTS_DIR / "tmp_embeddings.jsonl"


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


def run_align(args: list[str]) -> None:
    subprocess.run([str(PYTHON), str(ALIGN_ID), *args], cwd=str(BASE), check=True)


def run_extract(args: list[str]) -> None:
    subprocess.run([str(PYTHON), str(EXTRACT_ID), *args], cwd=str(BASE), check=True)


def prepare_embeddings(extra_args: list[str]) -> Path:
    run_align([*extra_args, "--output", str(TMP_ALIGN)])
    run_extract([*extra_args, "--input", str(TMP_ALIGN), "--output", str(TMP_EMBED)])
    return TMP_EMBED


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--t1-limit", type=int, default=50)
    parser.add_argument("--t2-limit", type=int, default=50)
    parser.add_argument("--t7-limit", type=int, default=1000)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for longueur in ["courte", "moyenne", "longue"]:
        extra = ["--longueur", longueur, "--limit", str(args.t1_limit)]
        if args.source:
            extra.extend(["--source", args.source])
        embed_path = prepare_embeddings(extra)
        before = count_lines(OUTPUT)
        run_calc(["--input", str(embed_path), "--echelle", "phrase", "--estimateur", "les_deux", "--longueur", longueur, "--limit", str(args.t1_limit)] + (["--source", args.source] if args.source else []))
        rows = load_new_rows(before)
        write_csv(rows, RESULTS_DIR / f"agg_T1_{longueur}.csv")

    for dominante in ["NOUN", "VERB", "ADJ", "mixte"]:
        extra = ["--dominante", dominante, "--limit", str(args.t2_limit)]
        if args.source:
            extra.extend(["--source", args.source])
        embed_path = prepare_embeddings(extra)
        before = count_lines(OUTPUT)
        run_calc(["--input", str(embed_path), "--echelle", "phrase", "--estimateur", "les_deux", "--dominante", dominante, "--limit", str(args.t2_limit)] + (["--source", args.source] if args.source else []))
        rows = load_new_rows(before)
        write_csv(rows, RESULTS_DIR / f"agg_T2_{dominante}.csv")

    extra = ["--limit", str(args.t7_limit)]
    if args.source:
        extra.extend(["--source", args.source])
    embed_path = prepare_embeddings(extra)
    before = count_lines(OUTPUT)
    run_calc(["--input", str(embed_path), "--echelle", "mot", "--estimateur", "mle", "--limit", str(args.t7_limit)] + (["--source", args.source] if args.source else []))
    rows = load_new_rows(before)
    write_csv(rows, RESULTS_DIR / "agg_T7_mle_mot.csv")


if __name__ == "__main__":
    main()
