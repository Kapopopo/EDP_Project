# Métriques d'entraînement — perte et perplexité

Date : 3 octobre 2026  
Branche : `P.A`

## Rappel

| Métrique | Objectif | Formule |
|----------|----------|---------|
| **Perte val** (`val_loss`) | **Minimiser** (vers 0, jamais atteint) | Entropie croisée |
| **Perplexité** | **Minimiser** | e^(perte) |

Plus la perte est basse, plus le modèle prédit bien le mot suivant sur les textes de validation.

## Progression

### Phase pilote (Wikipédia, 500 articles)

| Step | Perte val | Perplexité |
|------|-----------|------------|
| 10 | 8,27 | 3 919 |
| 510 | 5,70 | 299 |
| 2000 | **5,15** | **172** |

Checkpoint : `runs/pilot/best.pt`

### Phase juridique (1 000 articles filtrés)

Initialisée depuis `runs/pilot/best.pt`, corpus `data/legal/prepared/`.

| Step | Perte val | Perplexité |
|------|-----------|------------|
| 0 | 5,24 | 188 |
| 2000 | 4,73 | 114 |
| 4000 | 4,48 | 88 |
| 6250 | **4,34** | **77** |

Checkpoint : `runs/legal/best.pt`

### Comparaison sur validation juridique

| Modèle | Perte val | Perplexité |
|--------|-----------|------------|
| Pilote (`runs/pilot/best.pt`) | 5,19 | 179 |
| Juridique (`runs/legal/best.pt`) | **4,34** | **77** |

Amélioration : **−0,85** de perte (−16 %), perplexité divisée par ~2,3.

## Interprétation

- **Avant** (perte ~8) : quasi au hasard sur 4 096 tokens.
- **Maintenant** (perte **4,34**) : niveau **correct** pour un petit modèle (1,3 M paramètres).
- **Cible réaliste** sur Mac : perte 4–5, perplexité 50–150.
- **Zéro** : impossible et non souhaitable (surapprentissage).

Les textes générés restent incohérents : la perte mesure la prédiction statistique, pas la qualité juridique. Pour un usage cabinet, il faudra RAG + relecture avocat.

## Reprendre l'entraînement

```bash
# Continuer la phase juridique
.venv/bin/python -m llm_fr.train \
  --config runs/legal/config.json \
  --data data/legal/prepared \
  --output runs/legal --resume --stop-after 2000

# Tester une génération
.venv/bin/python -m llm_fr.generate runs/legal/best.pt \
  "Le secret professionnel de l'avocat impose" --count 80
```
