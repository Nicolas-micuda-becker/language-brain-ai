#!/usr/bin/env python3


from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge_rows(paths: list[Path], key: str) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            value = row.get(key)
            if value:
                merged[value] = row
    return merged


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def save_bar_plot(labels: list[str], values: list[float], title: str, ylabel: str, path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=color, edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results_dir = BASE_DIR / "tests" / "results"

    # T1: longueur
    t1_rows = merge_rows(
        [results_dir / "agg_T1_courte.csv", results_dir / "agg_T1_moyenne.csv", results_dir / "agg_T1_longue.csv"],
        "longueur",
    )
    length_order = ["courte", "moyenne", "longue"]
    length_two_nn = [to_float(t1_rows.get(label, {}).get("two_nn")) or 0.0 for label in length_order]
    length_mle = [to_float(t1_rows.get(label, {}).get("mle")) or 0.0 for label in length_order]
    save_bar_plot(length_order, length_two_nn, "ID Two-NN par longueur", "ID", FIG_DIR / "id_by_length_two_nn.png", "#4C78A8")
    save_bar_plot(length_order, length_mle, "ID MLE par longueur", "ID", FIG_DIR / "id_by_length_mle.png", "#F58518")

    # T2: POS dominant
    t2_rows = merge_rows(
        [
            results_dir / "agg_T2_NOUN.csv",
            results_dir / "agg_T2_VERB.csv",
            results_dir / "agg_T2_ADJ.csv",
            results_dir / "agg_T2_mixte.csv",
        ],
        "dominante",
    )
    pos_order = ["NOUN", "VERB", "ADJ", "mixte"]
    pos_two_nn = [to_float(t2_rows.get(label, {}).get("two_nn")) or 0.0 for label in pos_order]
    pos_mle = [to_float(t2_rows.get(label, {}).get("mle")) or 0.0 for label in pos_order]
    save_bar_plot(pos_order, pos_two_nn, "ID Two-NN par POS dominant", "ID", FIG_DIR / "id_by_pos_two_nn.png", "#54A24B")
    save_bar_plot(pos_order, pos_mle, "ID MLE par POS dominant", "ID", FIG_DIR / "id_by_pos_mle.png", "#E45756")

    # T6: validation A/B/C
    validation_path = OUTPUT_DIR / "statistical_validation.json"
    if validation_path.exists():
        data = json.loads(validation_path.read_text(encoding="utf-8"))
        group_labels = ["A", "B", "C"]
        two_nn = [data["two_nn"]["groups"][group]["mean"] for group in group_labels]
        mle = [data["mle"]["groups"][group]["mean"] for group in group_labels]
        save_bar_plot(group_labels, two_nn, "Validation A/B/C - Two-NN", "ID", FIG_DIR / "validation_ab_c_two_nn.png", "#72B7B2")
        save_bar_plot(group_labels, mle, "Validation A/B/C - MLE", "ID", FIG_DIR / "validation_ab_c_mle.png", "#B279A2")

    # T7: diagnostics MLE mot
    t7_csv = results_dir / "agg_T7_mle_mot.csv"
    t7_rows = read_csv(t7_csv)
    if t7_rows:
        mles = [row.get("mle") for row in t7_rows]
        twonns = [row.get("two_nn") for row in t7_rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(mles) + 1), [to_float(v) or 0.0 for v in mles], marker="o", label="MLE")
        if twonns:
            ax.plot(range(1, len(twonns) + 1), [to_float(v) or 0.0 for v in twonns], marker="o", label="Two-NN")
        ax.set_title("Diagnostics ID mot")
        ax.set_xlabel("Run")
        ax.set_ylabel("ID")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "mle_diagnostics.png", dpi=200)
        plt.close(fig)

    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
