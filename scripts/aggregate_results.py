#!/usr/bin/env python3
"""Aggregate ID/results JSONL and CSV outputs into a single CSV and optional .npz.

Usage:
  python scripts/aggregate_results.py

It looks for:
 - output/resultats_id.jsonl
 - tests/results/*.csv

Outputs:
 - output/aggregated_id_results.csv
 - output/aggregated_id_results.npz  (if numeric fields found)
"""
import json
import csv
import glob
import os
import numpy as np


def read_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def read_csv_rows(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def main():
    os.makedirs("output", exist_ok=True)
    rows = []

    # Read JSONL results
    for obj in read_jsonl("output/resultats_id.jsonl"):
        if isinstance(obj, dict):
            rows.append(obj)

    # Read CSVs under tests/results/
    for fn in sorted(glob.glob("tests/results/*.csv")):
        for r in read_csv_rows(fn):
            rows.append(r)

    if not rows:
        print("No results found to aggregate.")
        return

    # unify headers
    headers = []
    for r in rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)

    out_csv = "output/aggregated_id_results.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in headers})

    print(f"Wrote {len(rows)} rows to {out_csv}")

    # Try to save numeric arrays to .npz
    numeric_fields = [h for h in headers if any(h.lower().startswith(x) for x in ("id","two_nn","mle","score","value"))]
    arrays = {}
    for field in numeric_fields:
        vals = []
        for r in rows:
            v = r.get(field) if isinstance(r, dict) else None
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
        arrays[field] = np.array(vals, dtype=float)

    if arrays:
        npz_path = "output/aggregated_id_results.npz"
        np.savez_compressed(npz_path, **arrays)
        print(f"Saved numeric arrays to {npz_path}")


if __name__ == "__main__":
    main()
