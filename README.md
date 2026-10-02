# Préentraîner un petit modèle de langue français

Ce projet construit un **Transformer causal et un tokenizer BPE depuis zéro**, en Python avec PyTorch. Aucun poids de modèle existant n'est téléchargé. PyTorch fournit les tenseurs et les gradients ; `tokenizers` fournit l'algorithme BPE. Réécrire aussi ces bibliothèques serait un autre projet, moins adapté à une exécution efficace sur ordinateur personnel.

L'objectif de cette première étape est une chaîne de préentraînement compréhensible et vérifiable. **Ce n'est pas encore un assistant conversationnel, ni une source de réponses universellement fiables.** Un petit modèle entraîné localement n'aura pas les capacités d'un grand modèle généraliste. Une bonne perte de validation ne garantit ni des faits corrects ni le suivi des instructions.

Un premier essai CPU a déjà été exécuté dans cet espace de travail : 500 articles et dix mises à jour. Les résultats et la commande pour le reprendre figurent dans [docs/VERIFICATION.md](docs/VERIFICATION.md).

Le pilote a ensuite atteint son budget de 2 000 étapes. Pour poursuivre avec davantage de textes **sans perdre les poids appris**, consulter [docs/APPROFONDIR.md](docs/APPROFONDIR.md) : corpus élargi, conservation du tokenizer, nouvelle phase et comparaison avant/après.

Une troisième phase mélange maintenant Wikipédia et littérature française, avec un budget de 30 000 mises à jour et une surveillance RAM/disque. Consulter [docs/SESSION_LONGUE.md](docs/SESSION_LONGUE.md) pour suivre la session, l'arrêter proprement ou la reprendre. Les garde-fous de cette version nécessitent Linux.

## Installation

Python 3.10 ou plus récent. Depuis ce dossier, sous Linux :

```bash
# Crée un environnement isolé pour les bibliothèques du projet.
python -m venv .venv
# Active cet environnement dans le terminal actuel.
source .venv/bin/activate
# Installe une version CPU de PyTorch, sans bibliothèques CUDA volumineuses.
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
# Installe les autres dépendances, en conservant PyTorch déjà installé.
python -m pip install -r requirements.txt
# Vérifie le nettoyage, l'attention causale, l'apprentissage et la reprise.
python -m unittest discover -s tests -v
```

Pour un GPU NVIDIA, suivre le [sélecteur officiel PyTorch](https://pytorch.org/get-started/locally/) dans un environnement distinct, puis installer les dépendances. Le profil GPU doit être ajusté à la VRAM réelle ; sa présence n'est pas une garantie de compatibilité avec toutes les cartes. Le code prend actuellement en charge CPU et CUDA, pas Apple MPS.

## 1. Constituer les données

Chaque ligne du corpus est un document JSON autonome, en UTF-8 :

```json
{"text":"Un véritable document français suffisamment long…", "source":"URL ou référence de provenance", "license":"Conditions de réutilisation identifiées", "language":"fr"}
```

Les quatre champs sont obligatoires ; `title`, `date`, `revision` et d'autres métadonnées peuvent être ajoutés. La préparation rejette les documents de moins de 200 caractères. Éviter les documents gigantesques : l'encodage traite un document entier à la fois.

Une première source accessible est le snapshot français `20231101.fr` du dataset [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia). Sa fiche déclare CC-BY-SA-3.0 et GFDL ; conserver les références et examiner les conditions de réutilisation applicables. Wikipédia peut contenir des erreurs et ce snapshot date de novembre 2023.

```bash
# Place le cache de téléchargement dans le projet.
export HF_HOME="$PWD/data/cache/huggingface"
# Télécharge au maximum 10 000 articles français, progressivement.
python -m llm_fr.download --limit 10000
```

Le premier lot sert à vérifier le fonctionnement, **pas à obtenir une intelligence générale**. Le téléchargement peut transférer des blocs Parquet plus grands que les articles retenus ; prévoir réseau et disque. Aucun téléchargement ne se produit à l'import des modules. Le téléchargeur résout `main` en commit et l'enregistre dans chaque document ; pour reproduire une collecte ultérieure, fournir ce commit avec `--revision COMMIT_HUGGING_FACE`. Les empreintes locales identifient exactement les données effectivement utilisées.

Lire [docs/DONNEES.md](docs/DONNEES.md) avant d'augmenter fortement le volume : il décrit la revue de qualité et les limites du filtrage fourni.

## 2. Préparer le corpus et apprendre le tokenizer

```bash
# Nettoie et sépare les documents, apprend le BPE puis écrit les tokens sur disque.
python -m llm_fr.prepare data/raw/wikipedia_fr.jsonl data/prepared --vocab-size 4096
```

Le dossier de destination doit être **nouveau**. On conserve les accents et les paragraphes, on élimine les doublons exacts après normalisation et on réserve environ 5 % des documents pour la validation. Le tokenizer apprend uniquement sur les documents d'entraînement. Le BPE à base d'octets peut représenter des caractères français qu'il n'a jamais rencontrés.

La préparation produit :

| Fichier | Rôle |
|---|---|
| `train.jsonl`, `val.jsonl` | Documents nettoyés, séparés, avec leur provenance |
| `tokenizer.json` | Vocabulaire et règles BPE entraînés de zéro |
| `train.bin`, `val.bin` | Tokens en entiers non signés de 16 bits |
| `manifest.json` | Comptages, paramètres de séparation et empreintes SHA-256 |

Les données brutes, JSONL nettoyés, binaires et caches occupent simultanément de l'espace disque. Un milliard de tokens représente environ 2 Go pour le seul binaire. Les empreintes de déduplication restent en RAM : cette première version vise des corpus locaux modérés, pas des milliards de documents.

## 3. Mesurer avant d'entraîner longtemps

```bash
# Effectue dix mises à jour, avec le calendrier prévu pour l'expérience complète.
python -m llm_fr.train --config configs/cpu.json --stop-after 10
# Reprend avec les mêmes données, réglages, poids, optimiseur et états aléatoires.
python -m llm_fr.train --config configs/cpu.json --resume
```

Le profil CPU utilise 4 couches, une dimension de 128 et un contexte de 128 tokens, soit environ **1,33 million de paramètres avec 4 096 tokens de vocabulaire**. C'est un point de départ pédagogique volontairement petit. La taille exacte s'affiche au lancement. La taille du vocabulaire réellement apprise peut être plus petite que le maximum demandé.

Le débit `tokens_per_second` mesure la dernière mise à jour, hors validation et écriture des sauvegardes. Estimer le temps à partir de plusieurs mesures sur **ta machine** ; ne pas supposer qu'un entraînement massif terminera rapidement. La première validation vérifie également que les sauvegardes fonctionnent.

Lancer le profil NVIDIA, après installation adaptée et vérification de la mémoire :

```bash
# Utilise une expérience séparée pour le modèle GPU plus grand.
python -m llm_fr.train --config configs/gpu.json --output runs/gpu --stop-after 10
# Continue exactement la même expérience.
python -m llm_fr.train --config configs/gpu.json --output runs/gpu --resume
```

En cas de manque de mémoire, réduire `batch_size` puis `context`, et si nécessaire `width` ou `layers`, **dans une nouvelle expérience**. Augmenter `accumulation` maintient un lot effectif plus grand, mais augmente la durée de chaque mise à jour. Les réglages sont décrits dans [docs/COMPRENDRE.md](docs/COMPRENDRE.md).

## 4. Suivre la qualité et reprendre

`runs/cpu/metrics.jsonl` enregistre la perte d'entraînement, celle de validation, la perplexité, le taux, le nombre de tokens vus et le débit. Une perplexité plus faible indique une meilleure prédiction sur **ce corpus avec ce tokenizer** ; ne pas comparer directement des tokenizers différents. Elle est plafonnée numériquement à `exp(80)` dans le journal.

- `last.pt` contient la dernière évaluation sauvegardée et permet la reprise.
- `best.pt` contient la meilleure perte de validation observée.
- L'arrêt anticipé intervient après `patience` évaluations sans amélioration.
- `--stop-after N` termine proprement après N mises à jour supplémentaires et sauvegarde.
- Une fermeture brutale peut perdre les mises à jour depuis la dernière sauvegarde. Après interruption, `--resume` repart de `last.pt` ; une ligne du journal peut être répétée si l'interruption survient pendant l'écriture du checkpoint.

La reprise exige exactement les mêmes réglages et données. Elle ne prolonge pas une expérience déjà arrivée à `steps` ou à l'arrêt anticipé. Pour une nouvelle phase avec un autre budget ou davantage de données, utiliser `--init-from` dans un nouveau dossier, comme expliqué dans `docs/APPROFONDIR.md`. Planifier le budget avant de commencer, puis utiliser `--stop-after` pour découper les sessions. Les validations supplémentaires en fin de session peuvent modifier la sélection du meilleur checkpoint et la patience par rapport à une session continue.

Les fenêtres sont tirées aléatoirement avec remise ; `tokens_seen / tokens_du_corpus` mesure une exposition moyenne, **pas un nombre de passages exhaustifs**. Répéter indéfiniment un petit corpus favorise la mémorisation. Si la perte d'entraînement descend et celle de validation monte, revenir à `best.pt` et revoir les données ou la capacité du modèle.

## 5. Examiner une continuation

```bash
# Produit une continuation ; le préentraînement n'enseigne pas encore le dialogue.
python -m llm_fr.generate runs/cpu/best.pt "La langue française" --count 80
```

Les premières sorties peuvent être incohérentes. Le script recharge le tokenizer incorporé au checkpoint, afin d'éviter un vocabulaire incompatible. L'inférence est volontairement simple : CPU, contexte glissant, sans cache KV. Elle sert à examiner le préentraînement, pas à fournir une interface de chat optimisée.

## Tester en direct pendant le préentraînement

Dans un autre terminal ouvert dans le projet, lancer :

```bash
# Lance une session de saisie avec affichage progressif des sorties.
.venv/bin/python -m llm_fr.chat
```

Saisir un message puis Entrée. `/quitter`, Ctrl+C ou Ctrl+D ferment la session. Au démarrage, le chat choisit en priorité `runs/long/last.pt`, puis `runs/expanded/last.pt`, puis `runs/pilot/last.pt`, selon les sauvegardes présentes, et affiche le chemin choisi. Il recharge cette sauvegarde au début de chaque message : les nouveaux poids de cette phase sont pris en compte sans redémarrage. Si une autre phase est créée pendant une session, relancer le chat ou fournir son chemin avec `--checkpoint`. L'étape chargée s'affiche avec la réponse. Une réponse en cours conserve ses poids jusqu'à sa fin. Un verrou protège chaque dossier contre deux entraînements simultanés.

Le mode interactif utilise deux threads CPU et peut ralentir un entraînement exécuté en parallèle. Chaque message est indépendant, sans historique. Le format « Question / Réponse » est une amorce ; ce préentraînement encyclopédique n'a pas appris le dialogue. L'affichage en direct ne rend donc pas les réponses fiables ou cohérentes à lui seul. Utiliser `--checkpoint runs/pilot/best.pt` pour examiner le meilleur checkpoint plutôt que le plus récent.

## Où lire le code

Chaque ligne de code Python non vide porte une explication en français ou est une chaîne de documentation. Les configurations JSON n'acceptent pas les commentaires : tous leurs champs sont expliqués dans le guide.

| Fichier | Contenu |
|---|---|
| `llm_fr/__init__.py` | Déclare le paquet Python |
| `llm_fr/download.py` | Acquisition optionnelle et bornée de données françaises |
| `llm_fr/corpus.py` | Lecture, normalisation, déduplication et séparation |
| `llm_fr/prepare.py` | Entraînement du tokenizer, encodage et manifeste |
| `llm_fr/model.py` | Transformer causal écrit dans ce projet |
| `llm_fr/train.py` | Lots, optimisation, évaluation et sauvegarde/reprise |
| `llm_fr/generate.py` | Inspection de continuations de texte |
| `llm_fr/chat.py` | Session interactive, sorties progressives et rechargement des poids entre messages |
| `llm_fr/evaluate.py` | Comparaison de checkpoints sur des fenêtres de validation identiques |
| `configs/cpu.json`, `configs/gpu.json` | Profils de départ ajustables |
| `configs/continued_cpu.json` | Nouvelle phase CPU avec budget allongé et taux réduit |
| `tests/test_pipeline.py` | Tests de séparation, causalité, apprentissage et reprise |
| `tests/test_chat.py` | Vérification des accents dans le flux, du token de fin et des caractères de contrôle |
| `requirements.txt` | Plages de versions des bibliothèques nécessaires |
| `requirements-tested.txt` | Versions exactes installées lors de la vérification CPU sous Linux/Python 3.12 ; PyTorch CPU nécessite son index dédié |
| `.gitignore` | Exclut environnements, caches, données et checkpoints de Git |
| `docs/COMPRENDRE.md` | Explication des calculs et de tous les réglages |
| `docs/DONNEES.md` | Méthode de sélection et limites de fiabilité des données |
| `docs/VERIFICATION.md` | Tests effectués, résultats du pilote et commande de reprise |
| `docs/APPROFONDIR.md` | Extension du corpus, transfert des poids et mesures comparables |

Références techniques : [BPE et entraînement du tokenizer](https://huggingface.co/docs/tokenizers/quicktour), [attention causale optimisée PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [précision mixte PyTorch](https://docs.pytorch.org/docs/stable/notes/amp_examples.html). L'attention utilise le noyau optimisé disponible pour le matériel ; un gain de vitesse précis doit être mesuré.

## Notes de projet DATTY AI

DATTY AI : 3 modèles : 

1/ Compréhension -> Intent ->  SQL Engine -> PostgreSQL 

2/ Génération de texte 

3/Analyse -> Insights 


-> Dashboard Engine 




Sur le repo github : 

- Le Saas
- L'IA dévellopé

Sur le repo huggingFace : 

    - Les entrainements du modèles
