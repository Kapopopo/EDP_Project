# Vérification effectuée dans cet environnement

## Troisième phase : session longue Wikipédia + littérature

Une session longue a été démarrée en arrière-plan le **30 septembre 2026 à 15:05:37 UTC**, avec un maximum de 30 000 mises à jour, depuis le meilleur checkpoint de la deuxième phase au pas 3 000. Cet horaire est celui du lancement, pas celui d'une fin d'entraînement. Utiliser `.venv/bin/python -m llm_fr.status` pour connaître son état actuel ; le résultat peut changer après la rédaction de ce rapport.

La collecte contient 20 000 articles Wikipédia, dont 18 745 retenus pour l'entraînement et 1 034 pour la validation ; 221 sont rejetés par les filtres existants. Le flux d'entraînement contient 85 902 271 tokens. Les huit ouvrages français produisent 411 fragments d'entraînement (1 641 699 tokens) et 94 fragments réservés (324 046 tokens), tous ces derniers appartenant à Proust. L'audit `data/wiki_large/extension_audit.json` retrouve les 4 742 anciens documents d'entraînement et les 243 anciens documents de validation sans croisement exact ; aucun ouvrage littéraire n'est partagé entre les deux ensembles.

Le mélange tire 80 % des micro-lots dans Wikipédia et 20 % dans la littérature. Avant les nouvelles mises à jour, les pertes mesurées sont 4,6607 pour Wikipédia et 5,2514 pour la littérature ; leur moyenne pondérée est 4,7789. Ces valeurs sont enregistrées dans `runs/long/baseline.json` et constituent la référence propre à cette phase.

**Dix tests passent** après les modifications. Ils couvrent notamment les notices Gutenberg, la langue déclarée, les fragments de taille bornée, la séparation par ouvrage, les seuils de ressources, le verrou exclusif, la reprise bit à bit du mélange sur un réseau miniature CPU et la sauvegarde lors d'une demande STOP. Le suivi a confirmé que le verrou de la session longue était détenu après le retour du lanceur en arrière-plan.

Les commandes de suivi, d'arrêt propre et de reprise ainsi que les limites des garde-fous sont documentées dans `SESSION_LONGUE.md`. Le processus est borné par le budget, la patience et les réserves de ressources ; ce lancement ne prouve pas encore une qualité de réponse ni une autonomie générale.

## Préentraînement approfondi : deuxième phase terminée au pas 2 000

Après achèvement du pilote au pas 2 000, une nouvelle phase a été entraînée pendant **2 000 mises à jour supplémentaires** depuis `runs/pilot/best.pt`. Les poids n'ont pas été réinitialisés. Le modèle a désormais reçu 8 192 000 tokens au total, dont 4 096 000 dans cette nouvelle phase. La session est terminée ; aucun entraînement n'est laissé en cours par cette intervention.

Le corpus est passé de 500 à 5 000 articles collectés : 4 742 retenus pour l'entraînement, 243 pour la validation, 15 rejetés. Le vocabulaire de 4 096 tokens est conservé exactement. Le flux d'entraînement contient 30 136 978 tokens ; celui de validation, 1 621 585 tokens. L'audit confirme que les anciens documents restent dans leurs ensembles respectifs. Les défauts de qualité encore présents sont décrits dans `APPROFONDIR.md`.

Sur les fenêtres suivies pendant cette phase (64 lots de 2 séquences de 128 tokens), la perte descend de **5,2640 à 4,6266**, et la perplexité de **193,25 à 102,16**. Une seconde comparaison emploie 128 lots de 4 séquences, soit 65 536 tokens, avec une graine fixe différente :

| Validation utilisée | Perte avant | Perte après | Perplexité avant | Perplexité après |
|---|---:|---:|---:|---:|
| Corpus élargi | 5,2491 | 4,6939 | 190,39 | 109,27 |
| Ancien pilote | 5,1752 | 4,7793 | 176,83 | 119,02 |

Chaque ligne compare exactement les mêmes fenêtres, le même tokenizer et le même contexte pour les deux modèles. Cela montre une amélioration de prédiction sur ces deux échantillons réservés, sans régression mesurée sur l'ancien. Ce n'est pas une mesure de compréhension des questions ; les sorties intermédiaires examinées restent incohérentes.

Les six tests automatiques passent après extension du test de pipeline : transfert exact de tous les poids, maintien des ensembles, identité du tokenizer, rejet d'un vocabulaire incompatible, conservation de la provenance après reprise et comparaison reproductible. Le chat sans argument sélectionne désormais automatiquement `runs/expanded/last.pt` s'il existe.

Les résultats reproductibles se trouvent dans :

- `runs/expanded/baseline.json` : mesure avant la nouvelle phase.
- `runs/expanded/metrics.jsonl` : progression tous les 250 pas.
- `runs/expanded/comparison.json` : comparaison sur le corpus élargi.
- `runs/expanded/comparison_pilot.json` : contrôle sur l'ancienne validation.
- `runs/expanded/last.pt` et `best.pt` : poids sauvegardés au pas 2 000 de cette phase.
- `data/expanded/split_audit.json` et `quality_audit.json` : contrôles de séparation et indicateurs de qualité.

Pour poursuivre cette phase :

```bash
# Ajoute jusqu'à 1 000 mises à jour sans changer le calendrier de la phase.
.venv/bin/python -m llm_fr.train --config runs/expanded/config.json --data data/expanded --output runs/expanded --resume --stop-after 1000
```

Le budget global de cette phase est de 10 000 mises à jour, avec arrêt anticipé sur stagnation. Le guide complet est `APPROFONDIR.md`.

## Suite du 30 septembre 2026

Une session supplémentaire de 300 mises à jour a repris le pilote du pas 10 au pas 310. Le dernier checkpoint contient 634 880 tokens présentés depuis le départ. La perte de validation atteint 6,0109, contre 8,2736 au pas 10 ; la perplexité atteint environ 407,85. La session s'est terminée proprement et a enregistré `last.pt` et `best.pt`. Aucun entraînement ne reste lancé par cette session.

Le nouveau mode `.venv/bin/python -m llm_fr.chat` a été essayé avec le checkpoint du pas 310 et une saisie suivie de `/quitter`. Il produit des fragments en direct, mais le texte observé reste incohérent. Deux tests supplémentaires vérifient les accents coupés en tokens d'octets, la fin EOS et le retrait des caractères de contrôle du terminal.

## Essai initial

Le 24 septembre 2026, sous Linux/Python 3.12.10, avec PyTorch 2.14.0 CPU. Les versions installées sont conservées dans `requirements-tested.txt`. La machine expose environ 16 Go de RAM ; le profil essayé utilise quatre threads CPU. Aucun essai CUDA n'a été effectué.

## Résultats

- Les quatre tests de `python -m unittest discover -s tests -v` passent.
- Le test d'apprentissage exige une baisse de perte d'au moins moitié sur un motif miniature.
- Le test de reprise compare tous les poids après quatre mises à jour : résultat identique bit à bit sur CPU, y compris avec dropout et accumulation.
- `pip check` ne signale pas de dépendance cassée.
- La lecture distante a d'abord révélé un plantage de fermeture du streaming `datasets`/Arrow. Le téléchargeur utilise désormais une lecture Parquet synchrone ; un nouveau téléchargement de cinq articles s'est terminé avec le code de sortie 0 et a enregistré le commit résolu `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
- Un pilote réel de 500 articles Wikipédia français a été téléchargé et préparé : 470 documents d'entraînement, 30 de validation, aucun rejet ni doublon exact détecté.
- Le tokenizer pilote contient 4 096 tokens ; les flux contiennent 3 280 218 tokens d'entraînement et 264 842 tokens de validation.
- Le modèle CPU possède 1 331 968 paramètres. Dix mises à jour ont effectivement été exécutées, soit 20 480 tokens présentés.
- Au pas 10 : perte d'entraînement 8,2924, validation 8,2736, perplexité environ 3 919. La dernière mise à jour traite environ 7 005 tokens/seconde, hors validation et sauvegarde. Cette mesure courte ne prédit pas précisément un entraînement long.

Ces dix mises à jour valident le fonctionnement, **pas une compétence en français**. La perplexité est encore proche de celle d'un modèle peu informé sur un vocabulaire de cette taille. Aucune performance de réponse à des questions n'a été établie.

## Artefacts locaux et suite

Les artefacts volumineux sont exclus de Git mais présents dans l'espace de travail :

| Chemin | Contenu |
|---|---|
| `data/raw/wikipedia_fr_pilot.jsonl` | 500 articles bruts du pilote |
| `data/pilot/` | Documents séparés, tokenizer, binaires et manifeste |
| `runs/pilot/metrics.jsonl` | Mesure du pas 10 |
| `runs/pilot/last.pt` | État reprenable au pas 10 |
| `runs/pilot/best.pt` | Meilleur checkpoint de ce court essai |

Pour poursuivre ce pilote avec les réglages inchangés :

```bash
# Active les dépendances déjà installées.
source .venv/bin/activate
# Continue depuis le pas 10, jusqu'au budget du profil ou à l'arrêt anticipé.
python -m llm_fr.train --config configs/cpu.json --data data/pilot --output runs/pilot --resume
```

Pour un corpus plus grand et mieux contrôlé, utiliser les commandes du README avec de nouveaux dossiers. Le pilote présente des défauts d'extraction décrits dans `DONNEES.md` ; augmenter ses répétitions ne les corrige pas.
