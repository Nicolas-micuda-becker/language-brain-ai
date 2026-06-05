# Tests à réaliser — Observations sur l'ID

## 1. ID par catégorie POS dominante
Comparer l'ID moyenne des phrases VERB / NOUN / ADJ / ADV / mixte.
- Est-ce que les phrases à dominance verbale ont une ID plus haute que les nominales ?
- Les phrases ADJ sont-elles dans un espace plus "dense" ou plus "étalé" ?
- Hypothèse : VERB > NOUN > ADJ en termes d'ID (les verbes introduisent plus de relations syntaxiques)

## 2. ID par longueur de phrase
Comparer courte / moyenne / longue indépendamment du POS.
- Est-ce que l'ID croît avec la longueur ?
- Ou est-ce qu'une phrase courte mais dense en verbes a une ID comparable à une longue phrase nominale ?

## 3. Interaction longueur × POS
Croiser les deux variables : ex. VERB+courte vs VERB+longue, NOUN+courte vs NOUN+longue.
- Chercher si l'effet longueur est uniforme entre catégories ou s'il interagit avec le POS

## 4. ID par source / genre littéraire
Comparer l'ID globale par fichier source :
- Poésie (Baudelaire, Verlaine) vs Roman (Maupassant) vs Théâtre (Molière) vs Fable (La Fontaine)
- Hypothèse : la poésie compresse le sens → espace de plus faible dimension ? Ou au contraire plus dispersé ?

## 5. Corrélation ratio POS continu → ID
Pas juste la catégorie dominante, mais le ratio exact (0.0 à 1.0).
- Tracer la courbe ratio_VERB → ID, ratio_NOUN → ID, etc.
- Voir si la relation est linéaire, seuillée, ou non-monotone

## 6. ID à l'échelle mot vs phrase vs paragraphe
- Mot : nuage des embeddings de tous les mots d'une phrase → ID locale
- Phrase : nuage des vecteurs de phrases d'un paragraphe → ID méso
- Paragraphe : nuage des vecteurs de paragraphes d'un texte → ID globale
- Observer si l'échelle change les conclusions sur les POS

## 7. Évolution de l'ID dans un texte (tracking séquentiel)
Prendre un texte (ex: Le Horla) et calculer l'ID fenêtre par fenêtre (ex: 10 phrases glissantes).
- L'ID monte-t-elle dans les passages d'action ? Descend-elle dans les passages descriptifs ?
- Peut-on "lire" la structure narrative dans la courbe d'ID ?

## 8. Phrases mixtes vs dominantes
- Les phrases mixte ont-elles une ID intermédiaire entre VERB et NOUN ?
- Ou sont-elles dans un espace distinct ?

## 9. Moyenne vs médiane pour l'ID d'un document (→ Youssef)
Calculer l'ID d'un document par agrégation des ID de ses phrases.
- Comparer moyenne, médiane, max
- Quelle méthode est la plus stable entre documents du même genre ?

## 10. ADV : rôle modificateur
Les adverbes modifient verbes et adjectifs — observer si les phrases ADV se comportent plus comme les VERB ou comme les ADJ dans l'espace.
