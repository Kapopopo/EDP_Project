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
| refine 5000 | 3,68 | 40 |
| refine2 6000 | **3,64** | **38** |

Checkpoint final : **`runs/legal_medium_refine2/best.pt`** (4,2 M paramètres)

### Comparaison sur validation juridique

| Modèle | Paramètres | Perte val | Perplexité |
|--------|------------|-----------|------------|
| Pilote | 1,3 M | 5,19 | 179 |
| Petit juridique (`legal_refine3`) | 1,3 M | 4,09 | 60 |
| **Medium juridique (`legal_medium_refine2`)** | **4,2 M** | **3,64** | **38** |

Amélioration totale depuis le pilote : perte **5,19 → 3,64**, perplexité **179 → 38**.

## Interprétation

- **Avant** (perte ~8) : quasi au hasard sur 4 096 tokens.
- **Petit modèle** (1,3 M) : perte ~4,0 — plafond atteint.
- **Medium** (4,2 M) : perte **3,64** — proche de l'objectif ~3,0 ; sous 3,5 nécessiterait un modèle plus grand ou plus de données.
- **Zéro** : impossible et non souhaitable (surapprentissage).

Les textes générés restent incohérents : la perte mesure la prédiction statistique, pas la qualité juridique. Pour un usage cabinet, il faudra RAG + relecture avocat.

## Reprendre l'entraînement

```bash
# Reprendre le medium (meilleur modèle actuel)
.venv/bin/python -m llm_fr.train \
  --config runs/legal_medium_refine/config.json \
  --data data/legal/prepared \
  --output runs/legal_medium_refine --resume --stop-after 2000

# Tester une génération
.venv/bin/python -m llm_fr.generate runs/legal_medium_refine2/best.pt \
  "Le secret professionnel de l'avocat impose" --count 80

# Chat interactif (sélectionne automatiquement le meilleur checkpoint juridique)
.venv/bin/python -m llm_fr.chat
```
