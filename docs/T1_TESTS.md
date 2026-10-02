# Validation T1 — suite de tests sur Mac

Date : 3 octobre 2026  
Branche : `P.A`  
Plateforme : macOS (darwin), Python 3.13, PyTorch 2.14.1 CPU (ARM64)

## Environnement

```bash
/usr/local/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

Résultat `pip check` : aucune dépendance cassée.

## Compatibilité macOS

[`llm_fr/runtime.py`](../llm_fr/runtime.py) lit désormais la RAM via `/proc` sous Linux et via `ps` + `vm_stat` sous macOS. L'entraînement ne plante plus faute de `/proc`.

## Résultat des tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

| Fichier | Tests | Statut |
|---------|-------|--------|
| `tests/test_pipeline.py` | 4 | OK |
| `tests/test_resources_books.py` | 5 | OK |
| `tests/test_chat.py` | 2 | OK |

**Total : 11 tests, 0 échec** (dont `test_resources_snapshot`, nouvelle vérification portable macOS/Linux).

Couverture fonctionnelle :
- Nettoyage corpus et séparation train/val
- Causalité et apprentissage sur réseau miniature
- Pipeline complet et reprise bit à bit
- Livres Gutenberg, verrous, seuils RAM/disque
- Décodage UTF-8 progressif et caractères de contrôle

## Commande de reprise

```bash
.venv/bin/python -m unittest discover -s tests -v
```
