# Inventaire des sources juridiques — T1

Date : 3 octobre 2026  
Objectif : préparer la spécialisation juridique du modèle en T2.

## Sources retenues pour l'échantillon T1

| Source | Contenu | Licence | Volume T1 | Statut |
|--------|---------|---------|-----------|--------|
| Wikipédia FR filtrée (`llm_fr/download_legal.py`) | Articles encyclopédiques à thème juridique | CC-BY-SA-3.0 / GFDL | 1 000 docs (949 train / 51 val) | Collecté et préparé |
| Légifrance / data.gouv.fr | Codes, lois, décrets | Licence ouverte Etalab (vérifier par jeu) | 0 | À intégrer T2 |
| Judilibre (Cour de cassation) | Décisions de justice | Open data justice | 0 | À intégrer T2 |
| Modèles internes cabinet | Courriers, conclusions types | Interne / anonymisé | 0 | À créer avec avocats |

## Fiche type (à compléter par source)

Pour chaque nouvelle source, documenter :

1. **Provenance** — URL, organisme, date du dump
2. **Licence** — conditions de réutilisation vérifiées
3. **Volume** — nombre de documents, tokens estimés
4. **Qualité** — échantillon lu à la main, défauts observés
5. **Risques** — données personnelles, obsolescence, doublons
6. **Usage prévu** — préentraînement, RAG, modèles d'actes

## Échantillon T1 collecté

Commande :

```bash
.venv/bin/python -m llm_fr.download_legal --limit 1000 --output data/legal/raw/corpus.jsonl
.venv/bin/python -m llm_fr.prepare data/legal/raw/corpus.jsonl data/legal/prepared --tokenizer data/pilot/tokenizer.json
```

Le filtre par mots-clés (`droit`, `avocat`, `tribunal`, `jurisprudence`, etc.) oriente Wikipédia vers le domaine juridique sans prétendre remplacer Légifrance ou Judilibre.

Résultat de la préparation (`data/legal/prepared/manifest.json`) :
- 949 documents d'entraînement, 8 389 686 tokens
- 51 documents de validation, 474 090 tokens
- Tokenizer conservé depuis `data/pilot/tokenizer.json` (4 096 tokens)

## Limites assumées

- Wikipédia juridique ≠ textes de loi officiels ni jurisprudence certifiée
- Pas de garantie de véracité factuelle
- Le modèle devra citer des sources indexées (RAG) pour les faits sensibles
- Les modèles d'actes internes nécessitent anonymisation avant intégration

## Prochaines étapes T2

1. Télécharger un extrait Légifrance (Code civil, Code de procédure civile)
2. Évaluer Judilibre pour citations sourcées
3. Lancer `--init-from runs/pilot/best.pt` sur `data/legal/prepared`
