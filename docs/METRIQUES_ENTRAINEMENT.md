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
| 6250 | 4,34 | 77 |
| 10000 | 4,25 | 70 |
| refine 4000 | 4,18 | 65 |
| refine2 6000 | 4,13 | 62 |
| **refine3 8000** | **4,09** | **60** |

Checkpoint petit modèle : `runs/legal_refine3/best.pt` (perte 4,09)

### Modèle medium (4,2 M paramètres) — plus rapide vers ~3

| Step | Perte val | Perplexité |
|------|-----------|------------|
| 4500 | 3,95 | 52 |
| 8000 | **3,77** | **43** |
| refine 5000 | **3,69** | **40** |

Checkpoint final : **`runs/legal_medium_refine/best.pt`**

### Comparaison sur validation juridique

| Modèle | Perte val | Perplexité |
|--------|-----------|------------|
| Pilote (`runs/pilot/best.pt`) | 5,19 | 179 |
| Juridique v1 (`runs/legal/best.pt`) | 4,25 | 70 |
| **Juridique final (`runs/legal_refine3/best.pt`)** | **4,09** | **60** |

Amélioration totale depuis le pilote : **−1,10** de perte (−21 %), perplexité **179 → 60**.

## Interprétation

- **Avant** (perte ~8) : quasi au hasard sur 4 096 tokens.
- **Maintenant** (perte **4,09**) : objectif ~4,0 atteint pour un petit modèle (1,3 M paramètres).
- **Cible réaliste** sur Mac : perte 4–5, perplexité 50–150.
- **Zéro** : impossible et non souhaitable (surapprentissage).

Les textes générés restent incohérents : la perte mesure la prédiction statistique, pas la qualité juridique. Pour un usage cabinet, il faudra RAG + relecture avocat.

## Reprendre l'entraînement

```bash
# Reprendre depuis le meilleur modèle
.venv/bin/python -m llm_fr.train \
  --config runs/legal_refine3/config.json \
  --data data/legal/prepared \
  --output runs/legal_refine3 --resume --stop-after 2000

# Tester une génération
.venv/bin/python -m llm_fr.generate runs/legal_refine3/best.pt \
  "Le secret professionnel de l'avocat impose" --count 80
```
