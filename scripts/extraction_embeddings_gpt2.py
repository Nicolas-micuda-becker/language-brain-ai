import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).parent.parent
ALIGNMENT_PATH = BASE_DIR / "corpus" / "processed" / "corpus_alignement_gpt2.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "output" / "gpt2_embeddings.jsonl"
DEFAULT_MODEL = "asi/gpt-fr-cased-small"


def charger_alignements(path):
    return json.loads(path.read_text(encoding="utf-8"))


def filtrer_alignements(entrees, source=None, dominante=None, longueur=None, limit=None):
    if source:
        entrees = [e for e in entrees if e.get("source") == source]
    if dominante:
        entrees = [e for e in entrees if e.get("dominante") == dominante]
    if longueur:
        entrees = [e for e in entrees if e.get("longueur") == longueur]
    if limit is not None:
        entrees = entrees[:limit]
    return entrees


def verifier_tokenisation(entree, tokenizer):
    ids_attendus = [item["id"] for item in entree["bpe_tokens"]]
    ids_reels = tokenizer(entree["phrase"], add_special_tokens=False)["input_ids"]
    if ids_reels != ids_attendus:
        raise ValueError("Tokenisation differente du fichier d'alignement — regenerer l'etape 2.")


def extraire_phrase(entree, model, tokenizer, layer_index):
    verifier_tokenisation(entree, tokenizer)
    token_ids = [item["id"] for item in entree["bpe_tokens"]]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)

    if outputs.hidden_states is None:
        raise RuntimeError("Le modele n'a pas retourne de hidden states.")

    couche = outputs.hidden_states[layer_index][0]
    mots = []
    embeddings_mots = []

    for mot in entree.get("tokens", []):
        indices = mot.get("bpe_indices", [])
        if not indices:
            continue
        embedding = couche[indices].mean(dim=0).cpu().tolist()
        embeddings_mots.append(torch.tensor(embedding))
        mots.append({
            "index": mot.get("index"),
            "texte": mot.get("texte"),
            "lemme": mot.get("lemme"),
            "pos": mot.get("pos"),
            "bpe_indices": indices,
            "embedding": embedding,
        })

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
        "layer": layer_index,
        "embedding_dim": embedding_dim,
        "phrase_embedding": phrase_embedding,
        "words": mots,
        "nb_words_embedded": len(mots),
        "couverture_complete": entree.get("couverture_complete", False),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ALIGNMENT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--source", default=None)
    parser.add_argument("--dominante", default=None)
    parser.add_argument("--longueur", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    alignements = charger_alignements(args.input)
    alignements = filtrer_alignements(alignements, args.source, args.dominante, args.longueur, args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.output.open("w", encoding="utf-8") as f:
        for entree in alignements:
            f.write(json.dumps(extraire_phrase(entree, model, tokenizer, args.layer), ensure_ascii=False) + "\n")
            total += 1

    print(f"Phrases traitees : {total}")
    print(f"Sortie : {args.output}")


if __name__ == "__main__":
    main()
