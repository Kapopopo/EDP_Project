# Préparer des données utiles et traçables

Il n'existe pas un fichier contenant « toutes les connaissances fiables ». Le préentraînement compresse des régularités de textes dans des poids ; il ne constitue pas une base de données exhaustive et ne garantit pas la restitution fidèle des faits.

## Première collecte

Le téléchargeur fourni utilise uniquement [Wikimedia Wikipedia, snapshot français 20231101.fr](https://huggingface.co/datasets/wikimedia/wikipedia). Il conserve URL, titre, langue, snapshot et licence déclarée par la fiche du dataset. `--limit` borne les articles retenus ; leur ordre d'origine n'est pas un échantillonnage thématique représentatif. Un premier lot peut donc être biaisé. `--revision` permet de fixer un commit du dépôt.

Wikipédia apporte du français encyclopédique, mais peu de conversations et une couverture inégale des sujets. Compléter ensuite avec des textes français dont la provenance et les droits de réutilisation ont été vérifiés : ouvrages ouverts, supports pédagogiques, documentation et textes techniques. Ce sont des catégories à examiner, pas des corpus déjà collectés ou validés par ce projet.

L'inspection du pilote a notamment montré des dates manquantes dans le premier article (« né le … à … »). L'extraction des modèles et du balisage peut donc dégrader des phrases. Ce défaut n'est pas corrigé automatiquement : le pilote teste la chaîne technique et ne doit pas être considéré comme un corpus final impeccable.

## Ce que vérifie le code

- Présence de `text`, `source`, `license`, `language`, tous non vides.
- Langue **déclarée** égale à `fr` ; aucune détection linguistique automatique n'est encore implémentée.
- Normalisation Unicode NFC, espaces réduits, paragraphes conservés.
- Au moins 200 caractères et absence du caractère nul.
- Suppression des doublons exacts après nettoyage et conversion insensible à la casse.
- Séparation stable par empreinte de document, avant apprentissage du tokenizer.
- Empreintes des binaires et du tokenizer vérifiées avant entraînement.

## Ce qui demande encore une revue du corpus

Le filtre ne vérifie pas la véracité, les licences renseignées, les données personnelles, les doublons approximatifs, les traductions parallèles, le spam ou les textes générés. Deux versions légèrement différentes d'un même article peuvent se retrouver dans les deux ensembles. Avant un préentraînement conséquent, regrouper les versions et quasi-doublons, puis les affecter au même ensemble ou en garder une seule.

Pour chaque ajout de source, lire des échantillons variés et enregistrer : provenance, version/date, conditions de réutilisation, proportion de français, qualité rédactionnelle, domaines couverts, présence de contenus sensibles et duplication. Écarter les contenus personnels non nécessaires et les sources dont l'usage prévu n'est pas autorisé. Les textes provenant de sites reconnus restent susceptibles d'erreurs ou d'obsolescence.

Inspecter ensuite `manifest.json` : nombre de documents acceptés/rejetés, doublons et tokens par ensemble. Une très faible validation ou un corpus trop répétitif ne permettent pas de conclure à une bonne généralisation. La validation doit ressembler aux usages visés sans recopier l'entraînement.

## Augmenter le volume progressivement

1. Vérifier le pipeline et une courte exécution sur un corpus limité.
2. Mesurer le débit réel, la mémoire et la perte sur des documents réservés.
3. Élargir les sources et corriger les défauts de qualité observés.
4. Préparer un **nouveau** corpus et lancer une **nouvelle** expérience avec un budget de calcul explicite.
5. Conserver les données, configurations, empreintes, journaux et versions nécessaires à la comparaison.

Plus de répétitions sur les mêmes phrases ne remplace pas davantage de bons documents. Après cette étape, apprendre à suivre des instructions françaises puis évaluer les réponses sera un travail distinct. Pour des faits actualisés et des réponses sourcées, un mécanisme de recherche documentaire sera généralement nécessaire ; il n'est pas implémenté ici.
