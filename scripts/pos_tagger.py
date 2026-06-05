"""
POS-tagging pipeline pour corpus littéraire français (Gutenberg).
Input  : fichiers .txt dans corpus/raw/
Output : fichiers .json dans corpus/processed/ — une entrée par phrase
         avec texte, POS tags, ratios, longueur et catégorie dominante.
"""

import json
import re
from pathlib import Path
import spacy

nlp = spacy.load("fr_core_news_md")

# POS qu'on suit (tags universels spaCy)
POS_CIBLES = {"VERB", "NOUN", "ADJ", "ADV"}

RAW_DIR = Path(__file__).parent.parent / "corpus" / "raw"
OUT_DIR  = Path(__file__).parent.parent / "corpus" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def nettoyer_texte(texte: str) -> str:
    """Supprime les en-têtes Gutenberg et normalise les espaces."""
    # Supprime le header/footer Gutenberg standard
    for marqueur in ["*** START OF", "*** END OF", "Project Gutenberg"]:
        idx = texte.find(marqueur)
        if idx != -1:
            texte = texte[idx:]
            break
    # Normalise les sauts de ligne multiples
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    texte = re.sub(r"[ \t]+", " ", texte)
    return texte.strip()


def calculer_ratios_pos(doc_sent) -> dict:
    """Calcule le ratio de chaque POS sur les tokens lexicaux (sans ponctuation/espaces).
    AUX est fusionné dans VERB (auxiliaires = verbes fonctionnels)."""
    tokens_lex = [t for t in doc_sent if not t.is_punct and not t.is_space]
    total = len(tokens_lex)
    if total == 0:
        return {}
    compteurs = {pos: 0 for pos in POS_CIBLES}
    for t in tokens_lex:
        pos = "VERB" if t.pos_ == "AUX" else t.pos_
        if pos in POS_CIBLES:
            compteurs[pos] += 1
    return {pos: round(c / total, 4) for pos, c in compteurs.items()}


SEUILS_DOMINANCE = {"VERB": 0.25, "NOUN": 0.25, "ADJ": 0.15, "ADV": 0.15}

def categorie_dominante(ratios: dict) -> str:
    """Retourne le POS dominant si son ratio dépasse le seuil par catégorie, sinon 'mixte'.
    Seuil ADJ abaissé à 0.15 car les adjectifs dépassent rarement 25% en français naturel."""
    if not ratios:
        return "vide"
    dominant = max(ratios, key=ratios.get)
    seuil = SEUILS_DOMINANCE.get(dominant, 0.25)
    return dominant if ratios[dominant] >= seuil else "mixte"


def longueur_categorie(n_mots: int) -> str:
    if n_mots <= 8:
        return "courte"
    elif n_mots <= 18:
        return "moyenne"
    else:
        return "longue"


def tagger_fichier(chemin: Path) -> list[dict]:
    """Traite un fichier texte et retourne la liste des phrases annotées."""
    texte = chemin.read_text(encoding="utf-8", errors="ignore")
    texte = nettoyer_texte(texte)

    resultats = []
    # spaCy a une limite de taille — on découpe par paragraphes
    paragraphes = [p.strip() for p in texte.split("\n\n") if len(p.strip()) > 20]

    for para in paragraphes:
        doc = nlp(para)
        for sent in doc.sents:
            texte_phrase = sent.text.strip()
            if len(texte_phrase) < 10:
                continue
            # Filtre bruit : headers Gutenberg, marqueurs de dates, lignes parasites
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
            # Filtre : trop peu de tokens alphabétiques (ligne de symboles ou bruit)
            alpha = sum(1 for t in tokens_lex if t.is_alpha)
            if alpha / n_mots < 0.5:
                continue

            ratios = calculer_ratios_pos(sent)
            dom = categorie_dominante(ratios)

            # Détail token-level : texte + pos + lemme
            tokens_detail = [
                {"texte": t.text, "lemme": t.lemma_, "pos": t.pos_}
                for t in sent
                if not t.is_space
            ]

            resultats.append({
                "source": chemin.stem,
                "phrase": texte_phrase,
                "n_mots": n_mots,
                "longueur": longueur_categorie(n_mots),
                "ratios_pos": ratios,
                "dominante": dom,
                "tokens": tokens_detail,
            })

    return resultats


def main():
    fichiers = list(RAW_DIR.glob("*.txt"))
    if not fichiers:
        print(f"Aucun fichier .txt trouvé dans {RAW_DIR}")
        print("Dépose tes fichiers Gutenberg dans corpus/raw/ puis relance.")
        return

    print(f"{len(fichiers)} fichier(s) trouvé(s) dans corpus/raw/\n")

    toutes_phrases = []
    for fichier in fichiers:
        print(f"  Traitement : {fichier.name} ...", end=" ", flush=True)
        phrases = tagger_fichier(fichier)
        print(f"{len(phrases)} phrases")
        toutes_phrases.extend(phrases)

        # Sauvegarde par fichier source
        sortie = OUT_DIR / f"{fichier.stem}_tagged.json"
        sortie.write_text(json.dumps(phrases, ensure_ascii=False, indent=2), encoding="utf-8")

    # Sauvegarde globale
    sortie_globale = OUT_DIR / "corpus_complet.json"
    sortie_globale.write_text(json.dumps(toutes_phrases, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stats rapides
    print(f"\nTotal : {len(toutes_phrases)} phrases")
    from collections import Counter
    cats = Counter(p["dominante"] for p in toutes_phrases)
    lons = Counter(p["longueur"] for p in toutes_phrases)
    print("\nDistribution par catégorie dominante :")
    for cat, n in cats.most_common():
        print(f"  {cat:10s} : {n:5d} ({100*n/len(toutes_phrases):.1f}%)")
    print("\nDistribution par longueur :")
    for lon, n in lons.most_common():
        print(f"  {lon:10s} : {n:5d} ({100*n/len(toutes_phrases):.1f}%)")
    print(f"\nRésultats sauvegardés dans {OUT_DIR}")


if __name__ == "__main__":
    main()
