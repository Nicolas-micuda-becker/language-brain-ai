<<<<<<< HEAD
"""Extraction des embeddings GPT-2 (étape 3).

Lit le fichier d'alignement et produit un JSONL avec un embedding par mot
et un embedding global de phrase (moyenne des embeddings mots).
=======
"""
Usage:
    python scripts/extraction_embeddings_gpt2.py --limit 1
    python scripts/extraction_embeddings_gpt2.py --source maupassant_horla
>>>>>>> 5cdd0c0 (Etape 3)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


BASE_DIR = Path(__file__).parent.parent
ALIGNMENT_PATH = BASE_DIR / "corpus" / "processed" / "corpus_alignement_gpt2.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "output" / "gpt2_embeddings.jsonl"
DEFAULT_MODEL = "asi/gpt-fr-cased-small"


def charger_alignements(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def filtrer_alignements(
    entrees: list[dict[str, Any]],
    source: str | None = None,
    dominante: str | None = None,
    longueur: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    resultats = entrees
    if source:
<<<<<<< HEAD
        resultats = [entree for entree in resultats if entree.get("source") == source]
    if dominante:
        resultats = [entree for entree in resultats if entree.get("dominante") == dominante]
    if longueur:
        resultats = [entree for entree in resultats if entree.get("longueur") == longueur]
=======
        resultats = [entree for entree in resultats if entree["source"] == source]
    if dominante:
        resultats = [entree for entree in resultats if entree["dominante"] == dominante]
    if longueur:
        resultats = [entree for entree in resultats if entree["longueur"] == longueur]
>>>>>>> 5cdd0c0 (Etape 3)
    if limit is not None:
        resultats = resultats[:limit]
    return resultats


def verifier_tokenisation(entree: dict[str, Any], tokenizer) -> None:
    ids_attendus = [item["id"] for item in entree["bpe_tokens"]]
<<<<<<< HEAD
    ids_reels = tokenizer(entree["phrase"], add_special_tokens=False)["input_ids"]
    if ids_reels != ids_attendus:
        raise ValueError(
            "La tokenisation du tokenizer differe du fichier d'alignement. Regenerer l'etape 2 avec le meme modele."
=======
    tokens_attendus = [item["texte"] for item in entree["bpe_tokens"]]

    ids_reels = tokenizer(entree["phrase"], add_special_tokens=False)["input_ids"]
    tokens_reels = tokenizer.convert_ids_to_tokens(ids_reels)

    if ids_reels != ids_attendus or tokens_reels != tokens_attendus:
        raise ValueError(
            "La tokenisation du modele ne correspond pas au fichier d'alignement. "
            "Regenerer l'etape 2 avec le meme modele GPT-2."
>>>>>>> 5cdd0c0 (Etape 3)
        )


def moyenne_vecteurs(vecteurs: torch.Tensor) -> list[float]:
<<<<<<< HEAD
    return vecteurs.mean(dim=0).cpu().tolist()
=======
    return vecteurs.mean(dim=0).tolist()
>>>>>>> 5cdd0c0 (Etape 3)


def extraire_phrase(entree: dict[str, Any], model, tokenizer, layer_index: int) -> dict[str, Any]:
    verifier_tokenisation(entree, tokenizer)

    token_ids = [item["id"] for item in entree["bpe_tokens"]]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Le modele n'a pas retourne de hidden states.")

    couche = hidden_states[layer_index][0]

    mots = []
    embeddings_mots = []
<<<<<<< HEAD
    for mot in entree.get("tokens", []):
        indices = mot.get("bpe_indices", [])
        if not indices:
            continue
=======
    for mot in entree["tokens"]:
        indices = mot.get("bpe_indices", [])
        if not indices:
            continue

>>>>>>> 5cdd0c0 (Etape 3)
        vecteurs = couche[indices]
        embedding = moyenne_vecteurs(vecteurs)
        embeddings_mots.append(torch.tensor(embedding))
        mots.append(
            {
<<<<<<< HEAD
                "index": mot.get("index"),
                "texte": mot.get("texte"),
=======
                "index": mot["index"],
                "texte": mot["texte"],
>>>>>>> 5cdd0c0 (Etape 3)
                "lemme": mot.get("lemme"),
                "pos": mot.get("pos"),
                "bpe_indices": indices,
                "embedding": embedding,
            }
        )

    if embeddings_mots:
<<<<<<< HEAD
        phrase_embedding = torch.stack(embeddings_mots).mean(dim=0).cpu().tolist()
=======
        phrase_embedding = torch.stack(embeddings_mots).mean(dim=0).tolist()
>>>>>>> 5cdd0c0 (Etape 3)
        embedding_dim = len(embeddings_mots[0])
    else:
        phrase_embedding = []
        embedding_dim = 0

    return {
<<<<<<< HEAD
        "source": entree.get("source"),
        "phrase": entree.get("phrase"),
=======
        "source": entree["source"],
        "phrase": entree["phrase"],
>>>>>>> 5cdd0c0 (Etape 3)
        "n_mots": entree.get("n_mots"),
        "longueur": entree.get("longueur"),
        "dominante": entree.get("dominante"),
        "ratios_pos": entree.get("ratios_pos", {}),
<<<<<<< HEAD
        "model": getattr(tokenizer, "name_or_path", None),
=======
        "model": tokenizer.name_or_path,
>>>>>>> 5cdd0c0 (Etape 3)
        "layer": layer_index,
        "embedding_dim": embedding_dim,
        "phrase_embedding": phrase_embedding,
        "words": mots,
        "nb_words_embedded": len(mots),
        "couverture_complete": entree.get("couverture_complete", False),
    }


def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extraction des embeddings GPT-2")
    parser.add_argument("--input", type=Path, default=ALIGNMENT_PATH, help="Fichier d'alignement JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Fichier JSONL de sortie")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nom du modele GPT-2")
    parser.add_argument("--layer", type=int, default=-1, help="Indice de la couche a extraire")
    parser.add_argument("--source", default=None, help="Filtrer sur une source exacte")
    parser.add_argument("--dominante", default=None, help="Filtrer sur une dominante POS")
    parser.add_argument("--longueur", default=None, help="Filtrer sur une longueur")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de phrases")
    return parser.parse_args()


def main() -> None:
    arguments = parser_arguments()

    alignements = charger_alignements(arguments.input)
    alignements = filtrer_alignements(
        alignements,
        source=arguments.source,
        dominante=arguments.dominante,
        longueur=arguments.longueur,
        limit=arguments.limit,
    )

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    model = AutoModel.from_pretrained(arguments.model)
    model.eval()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with arguments.output.open("w", encoding="utf-8") as fichier_sortie:
        for entree in alignements:
            record = extraire_phrase(entree, model, tokenizer, arguments.layer)
            fichier_sortie.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

    print(f"Phrases traitees : {total}")
    print(f"Sortie : {arguments.output}")
    if total:
        print("Extraction terminee avec succes.")


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 5cdd0c0 (Etape 3)
