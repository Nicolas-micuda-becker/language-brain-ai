# Résultats & Notes

## Test 1 — ID par catégorie POS, échelle phrase
300 phrases Maupassant, Two-NN

VERB -> 14.31
NOUN -> 9.40
mixte -> 22.01
ADJ -> pas assez de phrases

ordre : mixte > VERB > NOUN

plus la phrase est grammaticalement variée, plus l'ID est haute
les phrases verbales occupent plus d'espace que les nominales

**pourquoi ?**
verbes = plus de relations syntaxiques à encoder (qui, quoi, quand, comment)
noms = plus statiques, espace plus compressé
cohérent avec l'idée que la syntaxe verbale est plus complexe

**limites**
source unique (Maupassant), 300 phrases seulement -> à confirmer sur corpus complet

---

## Test 2 — ID par longueur de phrase
900 phrases, 5 sources, Two-NN

courte  -> 2.10
moyenne -> 21.41
longue  -> 25.38

ordre : longue > moyenne >> courte

les phrases courtes ont une ID très basse
les phrases longues occupent beaucoup plus d'espace

**pourquoi ?**
phrases courtes = structures répétitives ("Il dit.", "Elle répond.", exclamations courtes) -> les vecteurs se ressemblent -> espace compressé
phrases longues = plus de contenu, plus de structures variées -> vecteurs plus dispersés
la longueur influence l'ID presque autant que la catégorie grammaticale

**à noter**
MLE reste élevé même pour les courtes (26) -> les deux estimateurs ne sont pas toujours d'accord sur les phrases courtes, signe d'instabilité à investiguer

---

## Test 3 — ID par source / genre littéraire
150 phrases par source, Two-NN

maupassant   -> 15.75  (roman, narration psychologique)
moliere      -> 19.56  (théâtre, dialogue)
baudelaire         -> 20.92  (poésie symboliste)
verlaine           -> 22.34  (poésie musicale)
lafontaine         -> 1.34   (fables)

ordre : verlaine > baudelaire > moliere > maupassant >> lafontaine

**pourquoi ?**
la poésie (Verlaine, Baudelaire) a une ID plus haute que la prose narrative -> espace plus riche et dispersé
le théâtre (Molière) est intermédiaire — dialogues variés mais structures conversationnelles répétitives
la fable (La Fontaine) a une ID très basse -> structures extrêmement formulaiques, morales répétitives, vocabulaire limité

**surprise : La Fontaine à 1.34**
les fables sont très courtes, très codifiées (animal + morale) -> GPT-2 les compresse dans un espace quasi unidimensionnel
pourrait être lié à la longueur moyenne très courte des phrases de fables

---

## Test 4 — échelle mot vs phrase (VERB)
172 phrases verbales, Two-NN

VERB échelle phrase -> 16.05
VERB échelle mot   -> 1.67 (MLE = infini, instable)

**pourquoi ?**
à l'échelle phrase : chaque vecteur représente une phrase entière -> variance réelle entre phrases -> ID cohérente
à l'échelle mot : les mêmes mots reviennent souvent ("je", "il", "était") -> vecteurs très proches malgré le contexte -> espace s'effondre
MLE infini = instabilité numérique quand beaucoup de voisins à distance quasi nulle

**conclusion**
l'échelle phrase est plus fiable et plus interprétable que l'échelle mot pour nos corpus
l'échelle mot nécessiterait de dédupliquer ou de filtrer les mots grammaticaux

---

## Test 5 — intensité verbale (ratio VERB croissant)
Two-NN, échelle phrase

ratio VERB ≥ 0.25 (n=175) -> 16.06
ratio VERB ≥ 0.35 (n=27)  -> 20.07
ratio VERB ≥ 0.45          -> pas assez de phrases

plus le ratio verbal est élevé, plus l'ID monte

**pourquoi ?**
une phrase très verbale accumule les actions, les temps, les relations -> GPT-2 a besoin de plus de dimensions
effet dose : l'ID n'est pas binaire (verbal / pas verbal), elle croît avec la densité verbale
cohérent avec le test 1 — renforce la conclusion sur les verbes
