import json
import re
from pathlib import Path
import spacy

nlp = spacy.load("fr_core_news_md")

POS_CIBLES = {"VERB", "NOUN", "ADJ", "ADV"}
SEUILS_DOMINANCE = {"VERB": 0.25, "NOUN": 0.25, "ADJ": 0.15, "ADV": 0.15}

RAW_DIR = Path(__file__).parent.parent / "corpus" / "raw"
OUT_DIR  = Path(__file__).parent.parent / "corpus" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def nettoyer_texte(texte):
    for marqueur in ["*** START OF", "*** END OF", "Project Gutenberg"]:
        idx = texte.find(marqueur)
        if idx != -1:
            texte = texte[idx:]
            break
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    texte = re.sub(r"[ \t]+", " ", texte)
    return texte.strip()


def calculer_ratios_pos(sent):
    tokens_lex = [t for t in sent if not t.is_punct and not t.is_space]
    total = len(tokens_lex)
    if total == 0:
        return {}
    compteurs = {pos: 0 for pos in POS_CIBLES}
    for t in tokens_lex:
        pos = "VERB" if t.pos_ == "AUX" else t.pos_
        if pos in POS_CIBLES:
            compteurs[pos] += 1
    return {pos: round(c / total, 4) for pos, c in compteurs.items()}


def categorie_dominante(ratios):
    if not ratios:
        return "vide"
    dominant = max(ratios, key=ratios.get)
    seuil = SEUILS_DOMINANCE.get(dominant, 0.25)
    return dominant if ratios[dominant] >= seuil else "mixte"


def longueur_categorie(n_mots):
    if n_mots <= 8:
        return "courte"
    elif n_mots <= 18:
        return "moyenne"
    return "longue"


def tagger_fichier(chemin):
    texte = chemin.read_text(encoding="utf-8", errors="ignore")
    texte = nettoyer_texte(texte)
    resultats = []
    paragraphes = [p.strip() for p in texte.split("\n\n") if len(p.strip()) > 20]

    for para in paragraphes:
        doc = nlp(para)
        for sent in doc.sents:
            texte_phrase = sent.text.strip()
            if len(texte_phrase) < 10:
                continue
            if "***" in texte_phrase:
                continue
            if re.match(r"^_\d+\s+\w+[\._]", texte_phrase):
                continue
            if re.match(r"^\*", texte_phrase):
                continue

            tokens_lex = [t for t in sent if not t.is_punct and not t.is_space]
            n_mots = len(tokens_lex)
            if n_mots < 3:
                continue
            alpha = sum(1 for t in tokens_lex if t.is_alpha)
            if alpha / n_mots < 0.5:
                continue

            ratios = calculer_ratios_pos(sent)
            tokens_detail = [
                {"texte": t.text, "lemme": t.lemma_, "pos": t.pos_}
                for t in sent if not t.is_space
            ]
            resultats.append({
                "source": chemin.stem,
                "phrase": texte_phrase,
                "n_mots": n_mots,
                "longueur": longueur_categorie(n_mots),
                "ratios_pos": ratios,
                "dominante": categorie_dominante(ratios),
                "tokens": tokens_detail,
            })

    return resultats


def main():
    fichiers = list(RAW_DIR.glob("*.txt"))
    if not fichiers:
        print(f"Aucun fichier .txt dans {RAW_DIR}")
        return

    print(f"{len(fichiers)} fichier(s) trouve(s)\n")
    toutes_phrases = []

    for fichier in fichiers:
        print(f"  {fichier.name} ...", end=" ", flush=True)
        phrases = tagger_fichier(fichier)
        print(f"{len(phrases)} phrases")
        toutes_phrases.extend(phrases)
        sortie = OUT_DIR / f"{fichier.stem}_tagged.json"
        sortie.write_text(json.dumps(phrases, ensure_ascii=False, indent=2), encoding="utf-8")

    sortie_globale = OUT_DIR / "corpus_complet.json"
    sortie_globale.write_text(json.dumps(toutes_phrases, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    print(f"\nTotal : {len(toutes_phrases)} phrases")
    for cat, n in Counter(p["dominante"] for p in toutes_phrases).most_common():
        print(f"  {cat:10s} : {n}")
    print(f"\nSauvegarde : {OUT_DIR}")


if __name__ == "__main__":
    main()
