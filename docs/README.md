# Documentation `bcs_determination` — Hub de référence

> **Point d'entrée de la documentation technique.** Ce dossier `docs/` est la
> source de vérité pour comprendre le code avant de le modifier. Tout agent (ou
> développeur) doit **lire le fichier pertinent ci-dessous AVANT chaque action**
> (lecture, écriture, refactor, exécution, entraînement).

## Protocole « lire avant d'agir »

1. **Toujours** commencer par ce fichier (`docs/README.md`) pour identifier le
   domaine concerné.
2. Ouvrir le(s) fichier(s) de référence correspondant(s) dans la table ci-dessous.
3. Vérifier les **conventions transverses** (plus bas) — elles s'appliquent à
   presque toutes les tâches.
4. Pour une reprise de travail sur la cascade BCS, lire aussi
   [BCS_CASCADE_PROGRESS.md](BCS_CASCADE_PROGRESS.md) (tracker inter-session).
5. En cas de contradiction entre la doc et le code, **le code fait foi** :
   corriger la doc dans le même changement.

## Table de routage

| Tu veux… | Lis d'abord |
|---|---|
| Comprendre le projet, la cascade, le package, l'inférence (CLI + pipeline) | [architecture.md](architecture.md) |
| Entraîner un modèle, toucher aux LightningModules / datamodules / configs | [training.md](training.md) |
| Modifier l'app web, la base SQLite, le frontend, le pré-chargement | [webapp.md](webapp.md) |
| Reprendre le chantier cascade espèce→race→silhouette→BCS | [BCS_CASCADE_PROGRESS.md](BCS_CASCADE_PROGRESS.md) |
| Vue produit / installation / setup SAM 3 | [../README.md](../README.md) |

## Carte rapide du dépôt

| Chemin | Rôle |
|---|---|
| `app.py` | App web FastAPI (UI interactive) — voir [webapp.md](webapp.md) |
| `inference.py` | CLI d'inférence (`--mode classify` / `--mode full`) — voir [architecture.md](architecture.md) |
| `train.py` | Point d'entrée d'entraînement Hydra — voir [training.md](training.md) |
| `src/bcs_pipeline/` | Package Python principal (toute la logique métier) |
| `configs/` | Configs Hydra (`config*.yaml`) |
| `scripts/` | Bootstrap, pré-chargement DB, entraînement BCS |
| `checkpoints/` | Poids entraînés + SAM 2/3 (**ignoré par git**) |
| `data/` | Datasets + `bcs_app.db` (**ignoré par git**) |
| `notebooks/` | Notebooks d'évaluation et de présentation |
| `experiments/` | Sorties d'entraînement Hydra/TensorBoard (**ignoré par git**) |

## Conventions transverses (à connaître avant presque toute tâche)

### Environnement & exécution
- Deux environnements possibles : **`uv`** (`.venv/`, Python 3.12, cf. README) ou
  **conda `bcs_analysis`** (Python 3.10). Les notebooks utilisent le kernel
  `bcs_analysis`.
- Ne **jamais** exécuter du code lourd (SAM 3, entraînement) sans GPU disponible
  sauf demande explicite ; forcer le CPU via `BCS_DEVICE=cpu` si besoin.
- Le package s'importe via `src/` : `sys.path.insert(0, "src")` (déjà fait dans
  `app.py`, `inference.py`, `train.py`, notebooks).

### Git
- `checkpoints/`, `data/`, `experiments/`, `*.ckpt`, `*.pt`, `*.pth`, `*.onnx`
  sont **ignorés** (`.gitignore`). Ne jamais tenter de committer des poids/données
  (28 Go de checkpoints — voir historique). Les modèles restent locaux.
- ⚠️ Le motif `data/` du `.gitignore` masque **aussi** `src/bcs_pipeline/data/`
  aux outils de recherche respectant gitignore. Ces datamodules sont pourtant
  bien suivis — utiliser une lecture directe si une recherche ne les trouve pas.
- Style de commits : **Conventional Commits en français** (`feat:`, `fix:`,
  `clean:`, `docs:`, `wip:`) — cf. `git log`.
- Ne jamais utiliser `--no-verify`, `--force`, `reset --hard` ; demander avant
  tout `git push`.

### Checkpoints — source de vérité unique
- **Tous** les chemins de checkpoints sont centralisés dans
  `src/bcs_pipeline/app_checkpoints.py` (fonctions `resolve_*_ckpt()`,
  `*_available()`, `describe_active_models()`). Ne pas coder de chemin en dur
  ailleurs ; passer par ce module.
- Emplacements attendus par l'app :
  - espèce → `checkpoints/classification/species/`
  - race chien → `checkpoints/classification/dog_breed/`
  - race chat → `checkpoints/classification/cat_breed/`
  - classifieur combiné (132) → `checkpoints/classification/vit_dogs_cats/`
  - segmentation → `checkpoints/segmentation/` (DeepLabV3, SAM 2.1, `sam3/`)
  - pose → `checkpoints/pose/yolo_best.pt`
  - BCS chat → `checkpoints/bcs_regression/cat/fold_*/` (rétrocompat : `fold_*` à
    plat = chat) ; BCS chien → `checkpoints/bcs_regression/dog/` (placeholder).

### Configuration (Hydra)
- Les hyperparamètres vivent dans `configs/*.yaml` ; overrides possibles en CLI.
- Toute nouvelle valeur de `dataset`/`model_name` doit être ajoutée au schéma
  `src/bcs_pipeline/utils/config_validation.py` (sinon la validation échoue).

### Pièges connus
- `predictions.json` de BCS peut contenir des **chemins absolus obsolètes** (autre
  machine / dataset déplacé) : reconstruire le chemin depuis le dossier courant.
- Les f-strings avec caractères `\u2501` dans `scripts/preload_db.py` déclenchent
  des warnings de lint en Python 3.10 (OK en ≥3.12) — connu, hors périmètre.
- `sam3/model_builder.py` fait `import pkg_resources` : garder `setuptools<81`.

## Maintenance de cette documentation
- Après tout changement structurel du code, **mettre à jour le fichier `docs/`
  concerné dans le même commit**.
- Garder cette doc factuelle et concise ; ne pas dupliquer le contenu du
  `README.md` produit (installation, setup) — y renvoyer.
