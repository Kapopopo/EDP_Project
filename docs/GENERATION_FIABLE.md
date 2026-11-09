# Génération fiable (Pack C)

Objectif : produire des mémos rédigés **ancrés** dans le contexte RAG, avec citations vérifiables, puis alimenter un jeu SFT (supervised fine-tuning) pour le modèle local.

## Mode citations strictes (`--strict-citations`)

Avec `python -m llm_fr.rag_chat --rediger --strict-citations` :

- Les extraits sont étiquetés `[S1]`, `[S2]`, … dans le prompt utilisateur.
- Le prompt système impose **une affirmation par puce**, suivie de `[S1]` ou `[S1][S2]`.
- Sans `--rediger`, le flag n’a pas d’effet (synthèse extractive inchangée).

Moteur de rédaction (`--backend`) :

- `auto` (défaut) : modèle local via **Ollama**, puis checkpoint local (`runs/…/best.pt`) si Ollama ne répond pas. **Aucune API distante n’est appelée**, même si une clé est définie.
- `ollama` : Ollama seul. Variables : `RAG_OLLAMA_MODEL` (nom affiché par `ollama list`), `RAG_OLLAMA_BASE_URL` (défaut `http://localhost:11434/v1`).
- `local` : checkpoint local seul.
- `api` : API compatible OpenAI **distante**, uniquement sur demande explicite (`--backend api`). Les extraits quittent la machine. Variables : `OPENAI_API_KEY` ou `RAG_LLM_API_KEY`, optionnellement `RAG_LLM_BASE_URL` et `RAG_LLM_MODEL`.

## Construire un dataset SFT depuis le RAG

1. **Interroger** le corpus (index production) avec une question métier réelle.
2. **Capturer** les `hits` renvoyés (texte, titre, source, domain) — même structure que `synthese_reply`.
3. **Rédiger** (avocat ou API en mode strict) un `target_memo` qui ne contient que des faits présents dans les hits, avec citations `[S1]…`.
4. **Enregistrer** une ligne JSONL par exemple.

### Schéma JSONL recommandé

```json
{
  "id": "uuid-ou-slug",
  "question": "Question client formulée en français",
  "hits": [
    {
      "text": "Extrait verbatim ou légèrement tronqué",
      "title": "Titre affiché",
      "source": "URL ou identifiant stable",
      "domain": "legifrance | judilibre | cabinet_curated"
    }
  ],
  "target_memo": "Mémo structuré avec puces et [S1][S2]…"
}
```

Champs optionnels utiles : `metadata.profession`, `metadata.intent`, `created_at`, `reviewer`.

### Bonnes pratiques

- Un hit = une source `[Sn]` ; ne pas réutiliser le même `n` pour deux documents différents dans un même exemple.
- Refuser les exemples où le `target_memo` cite un `[Sk]` absent des hits ou invente un article / arrêt.
- Mélanger questions courtes (déontologie) et longues (responsabilité + jurisprudence).
- Garder 10–20 % d’exemples « limites » (extraits insuffisants → memo qui le dit explicitement).

### Export depuis une session

Après une session synthèse, copier manuellement `question`, les hits affichés et le mémo validé ; ou automatiser via un petit script qui appelle `index.search` + `build_redaction_user_prompt(..., strict_citations=True)` pour générer la partie « instruction » côté modèle.

## Smoke eval

```bash
# Prompts + synthèse extractive (sans API)
python scripts/eval_redact_smoke.py

# + appel API si clé présente (sinon SKIP par cas)
python scripts/eval_redact_smoke.py --live-redact
```

Tests unitaires : `python -m unittest tests.test_llm_redact`.
