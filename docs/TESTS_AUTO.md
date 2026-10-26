# Tests automatiques (sans chat manuel)

## Une commande

```bash
./scripts/run_all_auto_tests.sh          # ~2–3 min (index chargé 1× pour golden)
./scripts/run_all_auto_tests.sh --full   # + 2000 questions smoke (~plus long)
```

Tu n’as **pas** besoin d’ouvrir `./scripts/rag.sh` pour valider le projet.

## Banque de 2000 questions

Fichier : `eval/question_bank_2000.jsonl`

Génération (combinaisons thèmes × formulations) :

```bash
.venv/bin/python scripts/generate_question_bank_2000.py
```

Test seul :

```bash
.venv/bin/python scripts/run_batch_eval.py --file eval/question_bank_2000.jsonl --min-rate 0.95
```

### Format JSONL (pour ton pote ou Excel)

| Champ | Obligatoire | Rôle |
|--------|-------------|------|
| `id` | oui | Identifiant unique |
| `query` | oui | Question |
| `theme` | recommandé | secret_pro, honoraires, etc. |
| `smoke` | oui pour masse | `true` = au moins 1 résultat, score top ≥ 0,08 |
| `expect_profession` | optionnel | `avocat`, `notaire`, … |
| `expect_intents` | optionnel | ex. `jurisprudence_strong` |
| `prefer_source` | optionnel | `judilibre` ou `legifrance` (≥2 en top5) |
| `prefer_any_of` | optionnel | comme golden cabinet |

Pour **renforcer** une question importante : copier une ligne depuis `eval/cabinet_queries.jsonl` ou `tests/fixtures/ranking_golden_queries.json`.

Export CSV (Excel) : ouvrir le JSONL dans LibreOffice / convertir avec `jq`.

## Golden strict (19 cas)

`scripts/run_eval.py` — seuil 60 % global, aujourd’hui 100 % attendu en dev.

## Rédaction IA

`scripts/eval_redact_smoke.py` — sans clé API (prompts + extractif).
