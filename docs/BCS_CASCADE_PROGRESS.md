# Suivi BCS — Pipeline cascade espèce → race → silhouette → BCS

> Fichier de reprise inter-session. Mets-le à jour après CHAQUE sous-tâche et
> CHAQUE entraînement (avant ET après lancement). Pour reprendre dans une nouvelle
> session : relire ce fichier en entier, puis vérifier la présence réelle des
> checkpoints sur disque (les statuts ci-dessous peuvent dater).

Légende : `[ ]` à faire · `[~]` en cours · `[x]` fait

## Objectif
1. Modèle ESPÈCE (chat/chien) → 2. Race routée (chien-only / chat-only) →
3. Segmentation (silhouette) → 4. BCS routé (modèle chat / modèle chien placeholder).
Pas d'embedding de race : la race agit via le routage par espèce.

## Tableau des entraînements
| Modèle        | Commande                                                        | Lancé | Abouti | Ckpt récupéré | Chemin                                   | Métrique | Date |
|---------------|-----------------------------------------------------------------|-------|--------|---------------|------------------------------------------|----------|------|
| species       | `python train.py --config-name config_species`                  | non   | non    | non           | checkpoints/classification/species/      | —        | —    |
| dog_breed     | `python train.py --config-name config_dog_breed`                | non   | non    | non           | checkpoints/classification/dog_breed/    | —        | —    |
| cat_breed     | `python train.py --config-name config_cat_breed`                | non   | non    | non           | checkpoints/classification/cat_breed/    | —        | —    |
| bcs_cat       | `python scripts/train_bcs_regression.py --species cat`          | non   | non    | non           | checkpoints/bcs_regression/cat/fold_*/   | —        | —    |
| bcs_dog       | (placeholder — en attente de données chien)                     | n/a   | n/a    | n/a           | checkpoints/bcs_regression/dog/          | —        | —    |

## Phase 0 — Fichier de suivi
- [x] Créer ce fichier `docs/BCS_CASCADE_PROGRESS.md`

## Phase 1 — Modèle espèce dédié (binaire)
- [x] `src/bcs_pipeline/data/species_classification_datamodule.py` (label 0=chien/1=chat)
- [x] `configs/config_species.yaml` (num_classes=2)
- [x] `train.py` : branche `dataset == "species"`
- [x] `src/bcs_pipeline/inference/species.py` : `load_species_model()`, `predict_species()`
- [ ] Entraînement species lancé puis abouti (maj tableau)

## Phase 2 — Deux classifieurs de race
- [x] `configs/config_dog_breed.yaml` (Stanford 120, datamodule existant)
- [x] `configs/config_cat_breed.yaml` (Oxford 12 chats) + `cat_breed_classification_datamodule.py` + branche train.py
- [x] Loaders noms de classes par espèce (`classification.py`)
- [ ] Entraînement dog_breed lancé puis abouti
- [ ] Entraînement cat_breed lancé puis abouti

## Phase 3 — BCS par espèce
- [x] `bcs.py` : `load_bcs_models()` registry {cat, dog}
- [x] `scripts/train_bcs_regression.py` : arg `--species` + `--data-dir`
- [x] Réorg checkpoints `bcs_regression/{cat,dog}/` (+ rétrocompat)
- [ ] Entraînement bcs_cat lancé puis abouti

## Phase 4 — Câblage cascade
- [x] `app_checkpoints.py` : nouveaux répertoires + fns de disponibilité
- [x] `inference_format.py` : cascade dans `run_core_inference` + champ `species`/`bcs_model_used`
- [x] `inference/pipeline.py` : cascade + params espèce/race
- [x] `app.py` : `_ensure_species/_dog_breed/_cat_breed/_bcs(species)` + routage
- [x] `inference.py` (CLI) : nouveaux args + résumé
- [x] `inference/__init__.py` : exports

## Phase 5 — Persistance + Frontend + Robustesse
- [x] `db.py` : colonne `predicted_species` + migration (+ tri + summary)
- [x] `static/js/app.js` + `templates/index.html` + `static/css/app.css` : badge espèce (overlay + historique), fallback chien BCS
- [x] Robustesse : masque absent (mask=None), chien sans modèle (placeholder), cache modèles CLI/app
- [x] `get_errors` sur tous les fichiers modifiés (seuls warnings préexistants restants dans preload_db.py f-strings \u2501, hors périmètre)

## Décisions
- Modèle espèce dédié ; 2 classifieurs de race dédiés ; 2 modèles BCS routés (pas d'embedding).
- BCS chien = placeholder (aucune donnée chien aujourd'hui).

## Hors scope
- Acquisition des données BCS chien (fournies plus tard).
- Réentraînement segmentation / pose.

## Notes / blocages
- Tout le code de la cascade est en place (Phases 1–5). Reste UNIQUEMENT les
  entraînements via `bash scripts/run_cascade_trainings.sh` (env conda bcs_analysis, py3.10).
- Bugs corrigés au 1er test : (1) config_validation.py `dataset` accepte désormais
  species/cat_breed ; (2) classification_module.py top_k=min(5,num_classes) (species=2 classes).
- Test OK : species ~99% val/acc, cat_breed ~69%, bcs_cat LOCO 11 folds tourne (SAM3 + tqdm).
- `preload_db.py` : warnings de lint préexistants (f-strings `\u2501` non supportées
  en Python 3.10, OK en ≥3.12) — hors périmètre, non modifiés.
- Rappel rapide des emplacements de checkpoints attendus par l'app :
  species → `checkpoints/classification/species/`,
  dog_breed → `checkpoints/classification/dog_breed/`,
  cat_breed → `checkpoints/classification/cat_breed/`,
  bcs_cat → `checkpoints/bcs_regression/cat/fold_*/` (rétrocompat: `fold_*` à plat = chat).
