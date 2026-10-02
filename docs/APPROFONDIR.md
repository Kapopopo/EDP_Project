# Continuer le préentraînement sans repartir de zéro

Le premier pilote s'est terminé après 2 000 mises à jour. Il a appris sur 500 articles et sa perte de validation finale est de 5,1492. Ce budget terminé explique pourquoi `--resume` seul ne le fait plus progresser.

La deuxième phase garde la même architecture CPU (1,33 million de paramètres, contexte de 128 tokens), les poids acquis et exactement le même tokenizer. Elle élargit la collecte à 5 000 articles du même snapshot Wikimedia. Ce choix reste compatible avec la mémoire de la machine ; il ne transforme pas le petit réseau en modèle généraliste.

**État actuel :** cette deuxième phase existe déjà dans `runs/expanded` et a effectué 2 000 mises à jour supplémentaires. Utiliser la commande `--resume` ci-dessous pour poursuivre ; ne pas relancer l'initialisation dans ce dossier. Les comparaisons avant/après sont dans `VERIFICATION.md`.

## Pourquoi conserver le tokenizer

Chaque ligne de la table d'embeddings correspond à un identifiant du vocabulaire. Réentraîner le BPE pourrait changer le sens de cet identifiant : les poids existants ne correspondraient plus aux textes. L'option `prepare --tokenizer` copie donc le vocabulaire existant octet pour octet et n'apprend aucune nouvelle fusion. Les caractères et mots inconnus restent représentables par leurs octets.

La séparation conserve la même graine et la même fraction de validation. Avec le même nettoyage et la même affectation par empreinte, un ancien document de validation reste réservé même s'il apparaît dans le nouveau corpus. Les variantes approximatives de documents ne sont toujours pas dédupliquées automatiquement : il ne s'agit pas d'une protection contre toutes les fuites possibles.

## Préparer une extension

Les chemins suivants correspondent à la nouvelle phase. Ne pas réexécuter la préparation si ces dossiers existent déjà ; utiliser les artefacts présents ou de nouveaux chemins.

```bash
# Télécharge un volume dix fois plus grand depuis la même version figée.
HF_HOME="$PWD/data/cache/huggingface" .venv/bin/python -m llm_fr.download --limit 5000 --revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa --output data/raw/wikipedia_fr_5000.jsonl
# Prépare les nouveaux textes avec le vocabulaire déjà appris.
.venv/bin/python -m llm_fr.prepare data/raw/wikipedia_fr_5000.jsonl data/expanded --tokenizer data/pilot/tokenizer.json
```

Les 5 000 articles sont les premiers de la collecte, pas un échantillon représentatif de tous les domaines. Leur extraction garde les limites signalées dans `DONNEES.md`. Cette extension augmente les textes disponibles, mais ne constitue pas une certification de leur véracité ou de leur qualité rédactionnelle.

La préparation effectuée a retenu 4 742 articles d'entraînement et 243 de validation ; 15 textes trop courts ont été rejetés. Cela représente 30 136 978 tokens d'entraînement et 1 621 585 de validation. L'audit de séparation dans `data/expanded/split_audit.json` retrouve les 470 anciens articles d'entraînement et les 30 anciens articles de validation dans leurs ensembles respectifs, sans croisement exact.

Un contrôle ciblé a trouvé des indices de phrases incomplètes dans 590 documents d'entraînement ; il ne détecte que quelques motifs et n'est pas une mesure exhaustive de qualité. Le rapport `data/expanded/quality_audit.json` conserve des exemples de sources à revoir. Ces documents n'ont pas été corrigés automatiquement ni présentés comme impeccables. Ne pas modifier les fichiers d'un corpus pendant un entraînement : préparer une nouvelle version pour toute correction.

## Démarrer une nouvelle phase ou reprendre celle qui existe

La première initialisation, dans un dossier nouveau, utilise :

```bash
# Transfère les poids et commence un nouveau calendrier d'optimisation.
.venv/bin/python -m llm_fr.train --config configs/continued_cpu.json --data data/expanded --output runs/expanded --init-from runs/pilot/best.pt --stop-after 2000
```

`--init-from` charge les poids uniquement. Un nouvel AdamW, un nouveau warmup et de nouveaux compteurs sont créés pour cette phase. Le tokenizer et l'architecture doivent rester identiques ; une incompatibilité provoque une erreur. L'ancien dossier est conservé. `origin` dans les checkpoints contient la provenance et les tokens présentés avant cette phase.

Une fois cette phase créée, employer **uniquement la reprise** :

```bash
# Reprend les poids, l'optimiseur et le calendrier exacts de la deuxième phase.
.venv/bin/python -m llm_fr.train --config runs/expanded/config.json --data data/expanded --output runs/expanded --resume --stop-after 1000
```

Chaque appel ajoute au maximum 1 000 mises à jour, dans la limite du budget global ou de l'arrêt anticipé. Retirer `--stop-after 1000` poursuit jusqu'à cette limite. Ne pas lancer simultanément deux entraînements dans le même dossier.

## Réglages de la deuxième phase

| Réglage | Valeur | Raison |
|---|---:|---|
| Budget maximal | 10 000 mises à jour | Autorise 20,48 millions de tokens présentés dans cette phase |
| Taux maximal | 0,00015 | Moitié du taux initial du pilote, pour adapter progressivement les poids existants |
| Warmup | 200 mises à jour | Laisse au nouvel optimiseur le temps d'estimer ses moments |
| Validation | Tous les 250 pas, 64 lots fixes | Échantillon plus grand pour suivre les variations de perte |
| Patience | 12 validations | Arrête après 12 mesures sans amélioration |
| Architecture et lot effectif | Inchangés | Conserve les poids et les besoins mémoire du petit modèle |

Ces valeurs sont un point de départ mesurable, pas des paramètres prouvés optimaux. Le calendrier prévoit davantage de calcul que le pilote ; mesurer la durée sur la machine réelle. `tokens_seen` compte les tokens de cette phase ; `total_tokens_seen` inclut les phases parentes. Les tirages restent aléatoires avec remise : ces compteurs ne garantissent pas que chaque document a été vu.

## Mesurer un progrès réel

Avant la première mise à jour, `baseline.json` mesure les poids parents sur les nouvelles fenêtres de validation. `best.pt` commence avec ces mêmes poids et n'est remplacé que lorsque la perte descend. Cela évite de déclarer meilleur un modèle qui aurait seulement moins régressé que les autres étapes de la nouvelle phase.

Pour une comparaison supplémentaire, le module `evaluate.py` impose le même tokenizer, la même longueur de contexte, les mêmes lots et la même graine pour tous les modèles :

```bash
# Compare ancien et nouveau modèles sur les mêmes fenêtres du corpus élargi.
.venv/bin/python -m llm_fr.evaluate runs/pilot/best.pt runs/expanded/best.pt --data data/expanded --batches 128 --output runs/expanded/comparison.json
# Contrôle aussi les progrès ou régressions sur l'ancienne validation.
.venv/bin/python -m llm_fr.evaluate runs/pilot/best.pt runs/expanded/best.pt --data data/pilot --batches 128 --output runs/expanded/comparison_pilot.json
```

Les rapports ne sont pas écrasés : choisir un autre nom pour une nouvelle comparaison. Le protocole est un échantillonnage de validation, pas un test indépendant de compréhension, un benchmark de raisonnement ou un examen exhaustif du corpus. Ne pas comparer directement les pertes de deux corpus différents.

## Tester les poids de la nouvelle phase en direct

```bash
# Recharge la dernière sauvegarde de cette phase entre les messages.
.venv/bin/python -m llm_fr.chat --checkpoint runs/expanded/last.pt
```

Pour observer le meilleur modèle validé, remplacer `last.pt` par `best.pt`. Le mode interactif ne lui enseigne pas le dialogue et ne mémorise pas les échanges. Apprendre à suivre des instructions et évaluer la pertinence des réponses demandera une étape distincte. Même une perte de préentraînement nettement meilleure ne permet pas d'affirmer que le modèle « comprend tout ».
