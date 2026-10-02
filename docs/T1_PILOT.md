# Entraînement pilote T1 — Mac CPU

Date : 3 octobre 2026  
Branche : `P.A`  
Config : [`configs/cpu.json`](../configs/cpu.json)

## Données

| Élément | Valeur |
|---------|--------|
| Source | Wikipédia FR `20231101.fr`, commit `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` |
| Articles collectés | 500 |
| Documents entraînement | 470 |
| Documents validation | 30 |
| Tokens entraînement | 3 280 218 |
| Tokens validation | 264 842 |
| Vocabulaire BPE | 4 096 |

## Entraînement

Modèle : 1 331 968 paramètres, contexte 128 tokens, CPU Mac ARM64.

| Step | Perte train | Perte val | Perplexité val |
|------|-------------|-----------|----------------|
| 10 | 8,29 | 8,27 | 3 919 |
| 100 | 7,03 | 6,99 | 1 091 |
| 200 | 6,54 | 6,39 | 598 |
| 300 | 5,95 | 6,03 | 417 |
| 400 | 5,91 | 5,84 | 343 |
| 500 | 5,76 | 5,72 | 304 |

Meilleur checkpoint : `runs/pilot/best.pt` (perplexité ~299 au step 510).

Durée totale : ~100 s pour les steps 10→510 (hors téléchargement et préparation).

## Génération (exemple)

Prompt : `Le secret professionnel de l'avocat`

```
Le secret professionnel de l'avocaton.
La « B'eau, des s'est pas est de la trus de l'un d'un vatation sur la pit pour l'histoire...
```

**Note :** le modèle apprend le français encyclopédique, pas encore le juridique. La baisse de perplexité mesure la prédiction sur le corpus Wikipédia, pas la qualité d'un conseil juridique.

## Fichiers produits

| Chemin | Rôle |
|--------|------|
| `data/raw/wikipedia_fr_pilot.jsonl` | Corpus brut |
| `data/pilot/` | Corpus préparé |
| `runs/pilot/metrics.jsonl` | Métriques |
| `runs/pilot/best.pt` | Meilleurs poids |
| `runs/pilot/training_500.log` | Journal complet |

## Reprise

```bash
.venv/bin/python -m llm_fr.train --config configs/cpu.json --data data/pilot --output runs/pilot --resume --stop-after 1000
```
