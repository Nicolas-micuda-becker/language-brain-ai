import argparse
import json
from pathlib import Path

from transformers import GPT2TokenizerFast

BASE_DIR = Path(__file__).parent.parent
CORPUS_PATH = BASE_DIR / "corpus" / "processed" / "corpus_complet.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "corpus" / "processed" / "corpus_alignement_gpt2.json"
DEFAULT_MODEL = "asi/gpt-fr-cased-small"


def charger_corpus(path):
    return json.loads(path.read_text(encoding="utf-8"))


def filtrer_entrees(entrees, source=None, dominante=None, longueur=None, limit=None):
    if source:
        entrees = [e for e in entrees if e["source"] == source]
    if dominante:
        entrees = [e for e in entrees if e["dominante"] == dominante]
    if longueur:
        entrees = [e for e in entrees if e["longueur"] == longueur]
    if limit is not None:
        entrees = entrees[:limit]
    return entrees


def reconstruire_spans_mots(phrase, tokens):
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
                raise ValueError(f"Impossible d'aligner {texte_token!r} dans {phrase!r}")
            texte_token = texte_compact
        fin = debut + len(texte_token)
        spans.append({
            "index": index,
            "texte": texte_token,
            "lemme": token.get("lemme"),
            "pos": token.get("pos"),
            "start": debut,
            "end": fin,
        })
        curseur = fin
    return spans


def aligner_phrase(entree, tokenizer):
    phrase = entree["phrase"]
    spans_mots = reconstruire_spans_mots(phrase, entree.get("tokens", []))
    encoding = tokenizer(phrase, add_special_tokens=False, return_offsets_mapping=True, return_attention_mask=False)
    bpe_ids = encoding["input_ids"]
    bpe_offsets = encoding["offset_mapping"]
    bpe_tokens = tokenizer.convert_ids_to_tokens(bpe_ids)

    bpe_items = [
        {"index": i, "id": bid, "texte": bt, "start": off[0], "end": off[1]}
        for i, (bid, bt, off) in enumerate(zip(bpe_ids, bpe_tokens, bpe_offsets))
    ]

    for mot in spans_mots:
        indices = [item["index"] for item in bpe_items if item["start"] < mot["end"] and item["end"] > mot["start"]]
        mot["bpe_indices"] = indices
        mot["bpe_tokens"] = [bpe_items[i]["texte"] for i in indices]
        mot["couverture_bpe"] = len(indices)

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
        "couverture_complete": all(mot["bpe_indices"] for mot in spans_mots),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default=None)
    parser.add_argument("--dominante", default=None)
    parser.add_argument("--longueur", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    corpus = charger_corpus(args.input)
    corpus = filtrer_entrees(corpus, args.source, args.dominante, args.longueur, args.limit)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    alignements = [aligner_phrase(e, tokenizer) for e in corpus]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(alignements, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Phrases alignees : {len(alignements)}")
    print(f"Sortie : {args.output}")


if __name__ == "__main__":
    main()
