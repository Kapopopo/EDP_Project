# Checklist livrables T1 — Fondations solides

Date : 3 octobre 2026  
Branche : `P.A`  
Commit : `2f11ec0`

## Statut global : T1 terminé

| # | Livrable | Critère | Statut | Preuve |
|---|----------|---------|--------|--------|
| 1 | Tests verts sur `P.A` | 10+ tests, push `origin/P.A` | **Fait** | [`docs/T1_TESTS.md`](T1_TESTS.md) — 11/11 OK |
| 2 | Compat Mac | Entraînement sans crash | **Fait** | [`llm_fr/runtime.py`](../llm_fr/runtime.py) + pilote 2000 steps |
| 3 | Métriques pilote | Perte/perplexité documentées | **Fait** | [`docs/T1_PILOT.md`](T1_PILOT.md), [`docs/METRIQUES_ENTRAINEMENT.md`](METRIQUES_ENTRAINEMENT.md) |
| 4 | Entretiens avocats | Guide + 2 comptes rendus | **Guide prêt** | [`docs/ENTRETIENS_T1.md`](ENTRETIENS_T1.md) — synthèses à remplir après entretiens réels |
| 5 | Architecture DATTY AI | Schéma MVP | **Fait** | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) |
| 6 | Corpus juridique | 500–2000 docs + fiches sources | **Fait** | 1000 docs, [`docs/SOURCES_JURIDIQUES.md`](SOURCES_JURIDIQUES.md), `data/legal/prepared/` |

## Artefacts clés

| Chemin | Description |
|--------|-------------|
| `runs/pilot/best.pt` | Pilote Wikipédia (step 2000, perte 5,15) |
| `runs/legal_medium_refine2/best.pt` | Meilleur modèle juridique (perte 3,64) |
| `data/legal/prepared/` | Corpus juridique prêt pour T2 |
| `llm_fr/download_legal.py` | Collecteur corpus juridique |

## Action restante (hors code)

- Conduire **2 entretiens avocats** et compléter les sections dans [`docs/ENTRETIENS_T1.md`](ENTRETIENS_T1.md)

## Commandes de vérification

```bash
git checkout P.A
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m llm_fr.generate runs/legal_medium_refine2/best.pt "Le secret professionnel" --count 40
```
