# Comprendre le préentraînement

## Ce que le modèle apprend

Un texte devient une suite d'entiers, les **tokens**. Un token peut être un morceau de mot, une ponctuation ou une partie d'un caractère. Le BPE apprend les regroupements fréquents à partir des seuls documents d'entraînement.

Pour les tokens `[10, 20, 30, 40]`, on fournit `[10, 20, 30]` au réseau et on lui demande de prédire `[20, 30, 40]`. Le masque causal empêche la position contenant `10` de regarder `20` ou `30`. Sans ce masque, le réseau pourrait tricher et la validation donnerait une fausse impression de réussite.

Le calcul suit les opérations de `model.py` :

1. **Embedding** : une table transforme chaque identifiant en vecteur de `width` nombres appris.
2. **Positions** : une deuxième table indique l'ordre des tokens. Le modèle ne sait pas traiter un contexte plus long que celui prévu.
3. **Normalisation** : `LayerNorm` stabilise les valeurs à l'entrée des transformations.
4. **Attention** : chaque position produit une requête Q, une clé K et une valeur V. Les scores QK divisés par la racine de la dimension d'une tête deviennent des probabilités après masquage causal et softmax. Leur moyenne pondérée des valeurs relie la position au texte précédent.
5. **Plusieurs têtes** : le vecteur est partagé en plusieurs sous-espaces ; `width` doit être divisible par `heads`.
6. **MLP** : deux couches linéaires et GELU transforment chaque position, avec une dimension intermédiaire quatre fois plus grande.
7. **Résidus** : chaque bloc ajoute sa transformation à son entrée, pour faciliter la circulation des gradients.
8. **Projection finale** : on réutilise la table des tokens pour produire un score par token possible. Ce partage économise des poids.
9. **Entropie croisée** : la perte est la moyenne de `-log(probabilité_du_vrai_token_suivant)`. Elle fournit un signal d'apprentissage, pas une mesure de vérité factuelle.
10. **Rétropropagation** : PyTorch calcule comment chaque poids contribue à l'erreur. AdamW modifie les poids pour réduire cette erreur.

L'architecture est compacte et classique : embeddings positionnels appris, LayerNorm, GELU, attention causale. Elle ne prétend pas reproduire toutes les optimisations des modèles industriels.

## Tous les champs de configuration

| Champ | Signification et effet |
|---|---|
| `seed` | Graine des poids et des tirages. Permet de reproduire une expérience dans le même environnement ; pas une garantie bit à bit entre différents GPU ou versions. |
| `device` | `cpu` ou `cuda`. Aucun basculement silencieux si CUDA manque. |
| `threads` | Nombre de threads CPU utilisés par PyTorch. Trop de threads peut ralentir un petit modèle. |
| `context` | Nombre de tokens fournis par fenêtre. Allonger le contexte augmente la mémoire et le calcul d'attention. |
| `width` | Dimension des vecteurs. L'augmenter augmente fortement les poids des couches denses. |
| `heads` | Nombre de têtes d'attention, diviseur exact de `width`. |
| `layers` | Nombre de blocs Transformer successifs. Plus de blocs signifie davantage de calcul et de paramètres. |
| `dropout` | Fraction de valeurs temporairement désactivées pendant l'entraînement, entre 0 inclus et 1 exclu. Désactivé en validation. |
| `batch_size` | Nombre de fenêtres traitées simultanément. Principal réglage pour réduire la mémoire des activations. |
| `accumulation` | Nombre de micro-lots avant une mise à jour. Les pertes sont divisées par ce nombre pour obtenir une moyenne correcte. |
| `steps` | Nombre maximal de mises à jour de poids de l'expérience complète. Ce n'est pas le nombre de documents. |
| `warmup` | Nombre de mises à jour de montée progressive du taux. Doit être inférieur à `steps` ; zéro désactive cette montée. |
| `learning_rate` | Taux maximal de mise à jour. Trop élevé : divergence ; trop faible : progrès lents. |
| `weight_decay` | Pénalisation AdamW des matrices de poids. Les biais et paramètres de normalisation sont exclus. |
| `eval_every` | Intervalle entre évaluations et sauvegardes, en mises à jour. |
| `eval_batches` | Nombre de lots réservés pour estimer la perte de validation. Les mêmes fenêtres sont réutilisées afin de comparer les étapes. |
| `patience` | Nombre d'évaluations successives sans baisse de validation avant arrêt anticipé. |

Les tokens prédits par mise à jour valent `batch_size × context × accumulation`. Avec le profil CPU : `2 × 128 × 8 = 2 048`. Les 2 000 mises à jour prévues représentent environ 4,1 millions de tokens présentés, éventuellement répétés. Ce budget initial n'est pas une promesse de maîtrise du français.

Après le warmup, le taux suit une décroissance cosinus jusqu'à 10 % du maximum. Les gradients sont bornés à une norme de 1. Sur GPU compatible, BF16 réduit la mémoire des activations ; sinon FP16 utilise un ajustement d'échelle. Les poids et états AdamW restent majoritairement en FP32. Sur CPU, on utilise FP32.

## Mémoire, validation et reproductibilité

Les flux de tokens sont ouverts avec `numpy.memmap` : le système charge les pages nécessaires depuis le disque. Seul le lot courant est copié en entiers 64 bits pour PyTorch. Les poids, gradients, moments AdamW et activations consomment aussi de la mémoire ; la taille du fichier de poids ne représente donc pas la mémoire d'entraînement.

Les documents sont concaténés avec un token de fin. Certaines fenêtres traversent plusieurs documents ; le modèle peut voir le document précédent après cette frontière. Le masque causal bloque le futur, mais n'isole pas chaque document dans une fenêtre. Cette convention simplifie le packing sans remplissage inutile.

Les fenêtres de validation sont un échantillon fixe avec remise, pas une évaluation exhaustive ni un benchmark de raisonnement. Le corpus de validation ne sert ni aux gradients ni à l'apprentissage du BPE. Il sert néanmoins à sélectionner les checkpoints : pour une évaluation finale indépendante, réserver plus tard un troisième ensemble de test et ne jamais s'en servir pour les réglages.

Un checkpoint contient le réseau, l'optimiseur, l'échelle AMP, les états aléatoires, les paramètres, le tokenizer, l'identité des données et les compteurs. L'écriture dans un fichier temporaire puis renommage protège le checkpoint précédent contre une écriture incomplète. Les versions des dépendances peuvent être conservées avec `python -m pip freeze > requirements-local.txt` dans l'environnement utilisé.

Chaque test de `tests/test_pipeline.py` a un objectif observable : accents préservés, aucun doublon exact entre ensembles, absence d'influence du futur, baisse de perte sur un motif simple et égalité exacte des poids CPU entre entraînement continu et reprise aux mêmes points d'évaluation. Réussir ces tests valide la mécanique, pas la qualité linguistique d'un modèle entraîné sur de vraies données.
