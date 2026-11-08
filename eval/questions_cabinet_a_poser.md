# Questions à poser à l’IA (cabinet avocat)

Copier-coller dans :
```bash
./scripts/ask_synthese.sh "votre question"
# ou
./scripts/rag.sh
# ou avec mémo rédigé :
./scripts/rag.sh --rediger --strict-citations
```

---

## Secret professionnel & procédure

1. Secret professionnel avocat — portée et exceptions
2. Correspondances avocat client peuvent-elles être versées au dossier pénal ?
3. Levée du secret professionnel avocat en enquête — rôle du bâtonnier
4. Secret professionnel avocat et réquisitions du parquet — jurisprudence Cour de cassation

## Honoraires & relation client

5. Convention honoraires avocat — informations obligatoires au client
6. Provision sur honoraires et facturation — décret 91-1197
7. Honoraires au résultat avocat — règles déontologiques
8. Contestation d’honoraires devant le bâtonnier — procédure

## Déontologie & organisation

9. Conflit d’intérêts avocat — obligations et refus de mission
10. Obligations déontologiques avocat décret 2023-552
11. Publicité avocat site internet — loyauté et dignité décret 91-1197
12. Mentions légales obligatoires site internet avocat

## RIN & barreau

13. Règlement intérieur national avocat — obligations principales
14. Organisation du barreau et déontologie — décret 91-1197

## Responsabilité & jurisprudence

15. Jurisprudence responsabilité civile avocat faute de conseil
16. Arrêt Cour de cassation responsabilité professionnelle avocat
17. Perte de chance et faute de l’avocat — jurisprudence

## Cas mixtes (test ranking)

18. Avocat secret défense nationale et jurisprudence cassation
19. Secret professionnel avocat vs secret médical — distinction
20. Obligations déontologiques notaire décret 2023-1297

---

## Export d’une réponse

```bash
EXPORT=exports/note.md ./scripts/export_synthese.sh "secret professionnel avocat"
```
