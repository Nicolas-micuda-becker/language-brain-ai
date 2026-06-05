"""
Alignement GPT-2 / corpus annoté pour l'etape 2 du projet.

Ce script lit le corpus deja annote (corpus_complet.json), recupere la
tokenisation GPT-2 avec les offsets caracteres, puis associe chaque mot du
corpus aux sous-mots BPE correspondants.

La sortie produit un JSON exploitable pour l'extraction des embeddings à
l'etape 3.

Usage:
    python scripts/alignement_gpt2.py
    python scripts/alignement_gpt2.py --source maupassant_horla --limit 50
    python scripts/alignement_gpt2.py --model antoiloui/gpt2-french
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import GPT2TokenizerFast


BASE_DIR = Path(__file__).parent.parent
CORPUS_PATH = BASE_DIR / "corpus" / "processed" / "corpus_complet.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "corpus" / "processed" / "corpus_alignement_gpt2.json"
DEFAULT_MODEL = "asi/gpt-fr-cased-small"


def charger_corpus(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def filtrer_entrees(
    entrees: list[dict[str, Any]],
    source: str | None = None,
    dominante: str | None = None,
    longueur: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    resultats = entrees
    if source:
        resultats = [entree for entree in resultats if entree["source"] == source]
    if dominante:
        resultats = [entree for entree in resultats if entree["dominante"] == dominante]
    if longueur:
        resultats = [entree for entree in resultats if entree["longueur"] == longueur]
    if limit is not None:
        resultats = resultats[:limit]
    return resultats


def reconstruire_spans_mots(phrase: str, tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruit les spans caracteres des mots du corpus dans la phrase."""
    spans = []
    curseur = 0

    for index, token in enumerate(tokens):
        texte_token = token["texte"]

        while curseur < len(phrase) and phrase[curseur].isspace():
            curseur += 1

        debut = phrase.find(texte_token, curseur)
        if debut == -1:
            texte_compact = texte_token.strip()
            if not texte_compact:
                continue
            debut = phrase.find(texte_compact, curseur)
            if debut == -1:
                raise ValueError(
                    f"Impossible d'aligner le token {texte_token!r} dans la phrase {phrase!r}"
                )
            texte_token = texte_compact

        fin = debut + len(texte_token)
        spans.append(
            {
                "index": index,
                "texte": texte_token,
                "lemme": token.get("lemme"),
                "pos": token.get("pos"),
                "start": debut,
                "end": fin,
            }
        )
        curseur = fin

    return spans


def aligner_phrase(entree: dict[str, Any], tokenizer: GPT2TokenizerFast) -> dict[str, Any]:
    phrase = entree["phrase"]
    tokens_mots = entree.get("tokens", [])
    spans_mots = reconstruire_spans_mots(phrase, tokens_mots)

    encoding = tokenizer(
        phrase,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    bpe_ids = encoding["input_ids"]
    bpe_offsets = encoding["offset_mapping"]
    bpe_tokens = tokenizer.convert_ids_to_tokens(bpe_ids)

    bpe_items = []
    for index, (bpe_id, bpe_token, offsets) in enumerate(zip(bpe_ids, bpe_tokens, bpe_offsets)):
        debut_bpe, fin_bpe = offsets
        bpe_items.append(
            {
                "index": index,
                "id": bpe_id,
                "texte": bpe_token,
                "start": debut_bpe,
                "end": fin_bpe,
            }
        )

    for mot in spans_mots:
        indices_bpe = [
            item["index"]
            for item in bpe_items
            if item["start"] < mot["end"] and item["end"] > mot["start"]
        ]
        mot["bpe_indices"] = indices_bpe
        mot["bpe_tokens"] = [bpe_items[i]["texte"] for i in indices_bpe]
        mot["couverture_bpe"] = len(indices_bpe)

    couverture_complete = all(mot["bpe_indices"] for mot in spans_mots)

    return {
        "source": entree["source"],
        "phrase": phrase,
        "n_mots": entree.get("n_mots"),
        "longueur": entree.get("longueur"),
        "dominante": entree.get("dominante"),
        "ratios_pos": entree.get("ratios_pos", {}),
        "tokens": spans_mots,
        "bpe_tokens": bpe_items,
        "nb_bpe": len(bpe_items),
        "couverture_complete": couverture_complete,
    }


def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alignement GPT-2 / mots du corpus")
    parser.add_argument("--input", type=Path, default=CORPUS_PATH, help="Chemin du corpus JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Chemin du fichier JSON d'alignement",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nom du tokenizer GPT-2")
    parser.add_argument("--source", default=None, help="Filtrer sur une source exacte")
    parser.add_argument("--dominante", default=None, help="Filtrer sur une dominante POS")
    parser.add_argument("--longueur", default=None, help="Filtrer sur une longueur")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de phrases")
    return parser.parse_args()


def main() -> None:
    arguments = parser_arguments()

    corpus = charger_corpus(arguments.input)
    corpus = filtrer_entrees(
        corpus,
        source=arguments.source,
        dominante=arguments.dominante,
        longueur=arguments.longueur,
        limit=arguments.limit,
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(arguments.model)

    alignements = []
    for entree in corpus:
        alignements.append(aligner_phrase(entree, tokenizer))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(alignements, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Phrases alignees : {len(alignements)}")
    print(f"Sortie : {arguments.output}")
    if alignements:
        premiere = alignements[0]
        print(f"Exemple : {premiere['phrase']}")
        print(f"- mots : {len(premiere['tokens'])}")
        print(f"- BPE : {premiere['nb_bpe']}")
        print(f"- couverture complete : {premiere['couverture_complete']}")


if __name__ == "__main__":
    main()