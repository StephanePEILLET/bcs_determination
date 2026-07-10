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
| Modèle        | Commande                                                        | Lancé | Abouti | Ckpt récupéré | Chemin                                   | Métrique          | Date       |
|---------------|-----------------------------------------------------------------|-------|--------|---------------|------------------------------------------|-------------------|------------|
| species       | `python train.py --config-name config_species`                  | oui   | oui    | oui           | checkpoints/classification/species/      | val/acc=1.00      | 2026-06-29 |
| dog_breed     | `python train.py --config-name config_dog_breed`                | oui   | oui    | oui           | checkpoints/classification/dog_breed/    | val/acc=0.89      | 2026-06-29 |
| cat_breed     | `python train.py --config-name config_cat_breed`                | oui   | oui    | oui           | checkpoints/classification/cat_breed/    | val/acc=0.95      | 2026-06-29 |
| bcs_cat       | `python scripts/train_bcs_regression.py --species cat --task classification` | oui | oui | oui | checkpoints/bcs_regression/cat/fold_*/ | acc=0.455 (=baseline) MAE=0.819 | 2026-07-10 |
| bcs_dog       | (placeholder — en attente de données chien)                     | n/a   | n/a    | n/a           | checkpoints/bcs_regression/dog/          | —                 | —          |

## Phase 0 — Fichier de suivi
- [x] Créer ce fichier `docs/BCS_CASCADE_PROGRESS.md`

## Phase 1 — Modèle espèce dédié (binaire)
- [x] `src/bcs_pipeline/data/species_classification_datamodule.py` (label 0=chien/1=chat)
- [x] `configs/config_species.yaml` (num_classes=2)
- [x] `train.py` : branche `dataset == "species"`
- [x] `src/bcs_pipeline/inference/species.py` : `load_species_model()`, `predict_species()`
- [x] Entraînement species lancé puis abouti (val/acc=1.00, 2026-06-29)

## Phase 2 — Deux classifieurs de race
- [x] `configs/config_dog_breed.yaml` (Stanford 120, datamodule existant)
- [x] `configs/config_cat_breed.yaml` (Oxford 12 chats) + `cat_breed_classification_datamodule.py` + branche train.py
- [x] Loaders noms de classes par espèce (`classification.py`)
- [x] Entraînement dog_breed lancé puis abouti (val/acc=0.89, 2026-06-29)
- [x] Entraînement cat_breed lancé puis abouti (val/acc=0.95, 2026-06-29)

## Phase 3 — BCS par espèce
- [x] `bcs.py` : `load_bcs_models()` registry {cat, dog}
- [x] `scripts/train_bcs_regression.py` : arg `--species` + `--data-dir`
- [x] Réorg checkpoints `bcs_regression/{cat,dog}/` (+ rétrocompat)
- [x] Entraînement bcs_cat régression (LOCO-CV 11 folds, MAE=0.821 RMSE=0.923, 2026-06-29)
- [x] **Reframe bcs_cat en classification ordinale** (scores discrets 4/5/6, `--task classification`,
  2026-07-10). Voir section « Reframe classification » ci-dessous.

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

## Reframe classification (2026-07-10)
- **Diagnostic régression** : le modèle BCS régression avait collapsé sur la moyenne
  (~4.82), prédiction quasi-constante indépendante de l'image (LV=DV identiques), MAE
  0.821 **pire** que la baseline « toujours la moyenne » (~0.744).
- **Décision** : reframer en **classification ordinale** sur les scores discrets présents
  (4/5/6). Sortie aval **inchangée** (`bcs` = valeur attendue `Σ pᵢ·scoreᵢ`, `category`),
  + nouveaux champs `bcs_class`, `confidence`, `probs`, `task`.
- **Code** : nouveau `bcs_classification_module.py` (`LitBCSClassification` +
  `BCSClassificationHead`, backbone ViT gelé partagé via `_BCSBackboneMixin`) ;
  `inference/bcs.py` détecte le task par le hparam `bcs_classes` (régression = rétrocompat
  totale) ; `train_bcs_regression.py` : `--task classification` (défaut), class-weights
  inverse-fréquence, `predictions.json` enrichi (accuracy, matrice de confusion, baselines).
- **Résultat honnête (80 epochs, 11 folds)** : **accuracy = 0.455 = baseline majoritaire**,
  MAE attendu 0.819 > baseline moyenne 0.744, matrice de confusion = **tout prédit en
  classe 4** (collapse persistant). Conclusion : les features du **ViT gelé** ne portent
  aucun signal de condition corporelle exploitable sur ces 22 images. Le reframe apporte
  l'**honnêteté** (confiance/probs + métriques + baselines) mais **pas** de gain de
  précision — le plafond est fixé par les données.
- **Prochaines pistes** (hors périmètre de ce reframe) : (a) dégeler partiellement le
  backbone / fine-tuner, (b) surtout **acquérir plus de données** (plus de chats, plage
  BCS complète 1–9, chiens). Cf. `bcs_dog` placeholder.
- Anciens checkpoints régression `cat/fold_*` déplacés vers
  `checkpoints/_trash_regression_cat_20260710/` (réversible).

## Notes / blocages
- **2026-06-29 : cascade complète et entraînée.** Les 4 modèles (species, dog_breed,
  cat_breed, bcs_cat) ont abouti et leurs checkpoints sont en place (voir tableau).
  Reste uniquement `bcs_dog`, bloqué en attente des données chien (hors scope).
- Le run `run_cascade_trainings.sh` a d'abord échoué sur `bcs_cat` faute du module
  `openpyxl` (moteur Excel de pandas pour lire le .xlsx OGR) → dépendance ajoutée
  dans `pyproject.toml` ; rerun OK.
- Checkpoints à plat legacy `checkpoints/bcs_regression/fold_OGR_*` (run mono-modèle
  du 2026-05-31, ~7 Go) rendus redondants par `bcs_regression/cat/fold_*`. Déplacés
  le 2026-07-10 vers `checkpoints/_trash_legacy_flat_bcs/` (corbeille locale) après
  vérification que `bcs_available('cat')` reste True. À vider quand tu veux :
  `rm -rf checkpoints/_trash_legacy_flat_bcs`.
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
