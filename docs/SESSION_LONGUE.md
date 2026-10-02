# Session longue : Wikipédia et littérature, avec RAM surveillée

Cette troisième phase repart des meilleurs poids de `runs/expanded`, après 3 000 mises à jour de la deuxième phase. Le tokenizer et l'architecture sont conservés. Le modèle reste un petit Transformer de 1,33 million de paramètres et de contexte 128 tokens : davantage de textes et de calcul ne suffisent pas à garantir un assistant autonome ou fiable.

## Données réellement sélectionnées

La collecte Wikipédia est bornée à 20 000 articles du snapshot français `20231101.fr`, commit `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Les anciennes affectations entraînement/validation sont conservées par empreinte. Les limites d'extraction déjà observées restent applicables ; ce corpus n'est pas une base de faits certifiés.

La préparation a retenu 18 745 articles d'entraînement et 1 034 articles de validation, soit respectivement 85 902 271 et 5 035 148 tokens. Les 221 autres documents ont été rejetés par les filtres. Le corpus littéraire apporte 1 641 699 tokens d'entraînement et 324 046 de validation. Environ 1,8 Gio de disque restaient disponibles après préparation, avant la croissance éventuelle des autres fichiers de la machine.

Le catalogue littéraire contient les éditions françaises suivantes, dont les fiches Project Gutenberg indiquent « Public domain in the USA ». Les textes originaux et leurs notices sont conservés localement, avec une empreinte d'édition dans chaque fragment. Cette mention du fournisseur n'est pas une certification universelle des droits de chaque usage.

| Ouvrage | Auteur | Fiche source |
|---|---|---|
| Le tour du monde en quatre-vingts jours | Jules Verne | [Gutenberg 800](https://www.gutenberg.org/ebooks/800) |
| La Mare au Diable | George Sand | [Gutenberg 23582](https://www.gutenberg.org/ebooks/23582) |
| Les misérables, tome I : Fantine | Victor Hugo | [Gutenberg 17489](https://www.gutenberg.org/ebooks/17489) |
| Le comte de Monte-Cristo, tome I | Alexandre Dumas et Auguste Maquet | [Gutenberg 17989](https://www.gutenberg.org/ebooks/17989) |
| Les trois mousquetaires | Alexandre Dumas et Auguste Maquet | [Gutenberg 13951](https://www.gutenberg.org/ebooks/13951) |
| Madame Bovary | Gustave Flaubert | [Gutenberg 14155](https://www.gutenberg.org/ebooks/14155) |
| Discours de la méthode | René Descartes | [Gutenberg 13846](https://www.gutenberg.org/ebooks/13846) |
| Du côté de chez Swann | Marcel Proust | [Gutenberg 2650](https://www.gutenberg.org/ebooks/2650) |

Les romans servent à apprendre des régularités de langue, des descriptions et des dialogues, pas à apprendre leurs événements fictifs comme des faits réels. Le livre de Descartes ajoute une prose argumentative ancienne. Le corpus ne représente pas à lui seul le français contemporain ni tous les domaines de connaissance.

`books.py` télécharge un livre à la fois, vérifie la langue déclarée, retire les notices situées hors des marqueurs Gutenberg et découpe le corps en fragments de 12 000 caractères au maximum. La clé `split_group` impose que tous les fragments d'un même ouvrage restent dans le même ensemble. Avec la graine 42 et le seuil 0,05, **Proust reste intégralement en validation** ; les sept autres ouvrages servent à l'entraînement. Les 94 fragments réservés ne sont donc pas 94 livres indépendants, et le style de Proust est une validation littéraire étroite.

## Mélange et budget

`configs/long_cpu.json` configure un maximum de **30 000 mises à jour**, soit **61,44 millions de tokens présentés** dans cette phase. C'est un budget maximal ; l'arrêt anticipé, un arrêt demandé ou les ressources disponibles peuvent raccourcir la session.

Chaque micro-lot est tiré dans Wikipédia avec probabilité 80 %, ou dans la littérature avec probabilité 20 %. Ce sont des proportions attendues, pas des quotas exacts à chaque étape. Les deux flux restent sur disque. Ce mélange évite que le petit corpus littéraire soit noyé dans le volume encyclopédique. Il peut toutefois répéter plusieurs fois les romans : surveiller leur perte de validation pour repérer la mémorisation.

Le taux maximal est 0,00012, avec 300 pas de warmup et décroissance cosinus. Le lot physique reste de deux fenêtres de 128 tokens, avec huit accumulations, soit 2 048 tokens par mise à jour. Trois threads CPU laissent davantage de marge aux autres applications. Les pertes Wikipédia et littérature sont enregistrées séparément ; le score de sélection du meilleur checkpoint est `0,8 × perte_Wikipédia + 0,2 × perte_littérature`. Cette moyenne de pertes n'est pas un score de compréhension.

Une validation a lieu tous les 250 pas. L'arrêt anticipé intervient après 16 validations sans amélioration du score combiné. Surveiller aussi les pertes individuelles : une amélioration moyenne peut masquer la régression d'un domaine.

## RAM et disque : protections effectives

Ces protections utilisent Linux (`/proc` et `fcntl`) :

- Les flux binaires sont lus avec `memmap`, sans charger tout le corpus en RAM.
- Les romans sont découpés avant la tokenisation ; chaque édition téléchargée est plafonnée à 16 Mio.
- Le téléchargement Wikipédia est plafonné à 400 Mio et conserve une réserve d'un Gio libre.
- La préparation vérifie une estimation conservative du disque nécessaire avant de créer les gros fichiers.
- L'entraînement mesure sa RAM résidente et la mémoire disponible **après chaque mise à jour complète**.
- Il demande un arrêt avec sauvegarde si sa RAM dépasse **1 800 Mio**, si Linux estime moins de **600 Mio disponibles**, ou si le disque descend sous **400 Mio libres**.
- Un verrou empêche deux entraînements de modifier simultanément le même dossier.

Les seuils sont une surveillance avec arrêt propre, pas une limite matérielle stricte. Une variation brutale, une allocation temporaire entre deux mesures, la consommation d'une autre application ou une panne disque peut encore provoquer un échec. La validation est sautée lorsqu'un arrêt de ressources est demandé, afin de réduire le travail avant la sauvegarde. Dans ce cas `last.pt` est mis à jour, mais `best.pt` reste le dernier modèle réellement validé.

## Suivre, tester, arrêter et reprendre

Depuis la racine du projet :

```bash
# Affiche l'activité réelle du verrou et la dernière mesure enregistrée.
.venv/bin/python -m llm_fr.status
# Observe le journal de la session démarrée en arrière-plan.
tail -f runs/long_training.log
# Teste en direct les derniers poids de cette phase.
.venv/bin/python -m llm_fr.chat --checkpoint runs/long/last.pt
# Demande un arrêt après la prochaine mise à jour, avec sauvegarde.
.venv/bin/python -m llm_fr.status --stop
```

Après une demande d'arrêt, relancer `status` pour vérifier que le verrou n'est plus détenu. Le journal doit confirmer la sauvegarde. Ctrl+C dans `tail -f` ferme seulement le suivi du journal ; cela n'arrête pas un entraînement lancé en arrière-plan.

Pour reprendre une session effectivement arrêtée :

```bash
# Reprend l'état exact de la session longue dans le terminal courant.
.venv/bin/python -m llm_fr.train --config runs/long/config.json --data data/wiki_large --output runs/long --resume
```

La session lancée en arrière-plan ne dépend pas de la saisie du chat, mais elle nécessite que l'ordinateur reste allumé et actif. Aucun redémarrage automatique du système n'est configuré. Après extinction ou arrêt brutal, reprendre depuis `last.pt` ; les pas depuis la dernière sauvegarde peuvent être perdus.

## Reproduire les données dans de nouveaux dossiers

Les commandes suivantes servent à reproduire la collecte ; ne pas écraser les artefacts déjà présents :

```bash
# Collecte bornée du snapshot Wikipédia figé.
HF_HOME="$PWD/data/cache/huggingface" .venv/bin/python -m llm_fr.download --limit 20000 --max-mb 400 --revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa --output data/raw/wikipedia_fr_20000.jsonl
# Collecte le catalogue français sélectionné et conserve les originaux.
.venv/bin/python -m llm_fr.books
# Encode Wikipédia avec le vocabulaire appris lors du pilote.
.venv/bin/python -m llm_fr.prepare data/raw/wikipedia_fr_20000.jsonl data/wiki_large --tokenizer data/pilot/tokenizer.json
# Encode la littérature avec exactement le même vocabulaire.
.venv/bin/python -m llm_fr.prepare data/raw/books_fr.jsonl data/literature --tokenizer data/pilot/tokenizer.json
```

Pour une création initiale de cette phase dans un dossier vide, la commande d'apprentissage utilise `--init-from runs/expanded/best.pt` avec `--config configs/long_cpu.json --data data/wiki_large --output runs/long`. Une fois la phase créée, utiliser `--resume` et non `--init-from`.

## Nouveaux fichiers

| Fichier | Fonction |
|---|---|
| `configs/books.json` | Catalogue des huit éditions françaises, identifiant, titre et auteur |
| `configs/long_cpu.json` | Budget, mélange des domaines et seuils de surveillance |
| `llm_fr/books.py` | Collecte bornée, notices retirées du texte d'apprentissage, découpage traçable |
| `llm_fr/runtime.py` | Verrou exclusif, lecture RAM/disque et motifs d'arrêt |
| `llm_fr/status.py` | Suivi et demande d'arrêt sans charger le modèle |
| `tests/test_resources_books.py` | Langue, notices, taille des fragments, séparation par ouvrage et garde-fous |

Les nouveaux champs sont `literature_data` (dossier du corpus littéraire), `literature_probability` (probabilité de sélectionner un lot littéraire), `max_rss_mb` (RAM résidente surveillée), `min_available_mb` (réserve globale de RAM) et `min_disk_mb` (réserve disque). Malgré le suffixe `mb`, les mesures sont en **Mio**, soit 1 048 576 octets. Les autres champs restent expliqués dans `COMPRENDRE.md`.
