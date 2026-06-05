"""Extraction des embeddings GPT-2 (étape 3).

Lit le fichier d'alignement et produit un JSONL avec un embedding par mot
et un embedding global de phrase (moyenne des embeddings mots).
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
        resultats = [entree for entree in resultats if entree.get("source") == source]
    if dominante:
        resultats = [entree for entree in resultats if entree.get("dominante") == dominante]
    if longueur:
        resultats = [entree for entree in resultats if entree.get("longueur") == longueur]
    if limit is not None:
        resultats = resultats[:limit]
    return resultats


def verifier_tokenisation(entree: dict[str, Any], tokenizer) -> None:
    ids_attendus = [item["id"] for item in entree["bpe_tokens"]]
    ids_reels = tokenizer(entree["phrase"], add_special_tokens=False)["input_ids"]
    if ids_reels != ids_attendus:
        raise ValueError(
            "La tokenisation du tokenizer differe du fichier d'alignement. Regenerer l'etape 2 avec le meme modele."
        )


def moyenne_vecteurs(vecteurs: torch.Tensor) -> list[float]:
    return vecteurs.mean(dim=0).cpu().tolist()


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
    for mot in entree.get("tokens", []):
        indices = mot.get("bpe_indices", [])
        if not indices:
            continue
        vecteurs = couche[indices]
        embedding = moyenne_vecteurs(vecteurs)
        embeddings_mots.append(torch.tensor(embedding))
        mots.append(
            {
                "index": mot.get("index"),
                "texte": mot.get("texte"),
                "lemme": mot.get("lemme"),
                "pos": mot.get("pos"),
                "bpe_indices": indices,
                "embedding": embedding,
            }
        )

    if embeddings_mots:
        phrase_embedding = torch.stack(embeddings_mots).mean(dim=0).cpu().tolist()
        embedding_dim = len(embeddings_mots[0])
    else:
        phrase_embedding = []
        embedding_dim = 0

    return {
        "source": entree.get("source"),
        "phrase": entree.get("phrase"),
        "n_mots": entree.get("n_mots"),
        "longueur": entree.get("longueur"),
        "dominante": entree.get("dominante"),
        "ratios_pos": entree.get("ratios_pos", {}),
        "model": getattr(tokenizer, "name_or_path", None),
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
    main()
