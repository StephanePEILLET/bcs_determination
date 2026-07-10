# Application web & persistance

> `app.py` (FastAPI/Uvicorn), la base SQLite (`db.py`), le frontend
> (`templates/` + `static/`) et le pré-chargement (`scripts/preload_db.py`). Lire
> ce fichier avant de modifier l'API web, le schéma de base, ou l'interface.

## `app.py` — application FastAPI

Interface web interactive répliquant le widget `combined_inference_overlay.ipynb`.
Servie par **Uvicorn** ; port par défaut `DEFAULT_PORT` (surchargeable via `PORT`).

- **Lazy loading** : les modèles sont chargés en mémoire au **premier** appel
  d'inférence (helpers `_ensure_classifier`, `_ensure_segmenter`, `_ensure_pose`,
  `_ensure_species`, `_ensure_dog_breed`, `_ensure_cat_breed`, `_ensure_bcs`).
  Le premier run est lent, les suivants instantanés. Cache dans un dict `_MODELS`.
- **Chemins de checkpoints** : importés depuis `app_checkpoints.py` (jamais en dur).
  Le BCS et les modèles espèce/race sont **optionnels** : l'app fonctionne sans eux.
- **Cascade** : `_run_inference_on_image` appelle `run_core_inference` (espèce →
  race routée → pose? → segmentation) puis le BCS (routé par espèce, masque de
  segmentation), et enfin `format_inference_result(...)`.

### Endpoints

| Méthode & route | Rôle |
|---|---|
| `GET /` | Page unique (Jinja2 `templates/index.html`) |
| `GET /api/datasets` | Groupes/fichiers par dataset (pour l'UI) |
| `GET /api/images` | Liste d'images filtrable |
| `GET /api/thumbnail/{dataset}/{group}/{filepath:path}` | Miniature d'image |
| `POST /api/inference` | Inférence sur une image de dataset |
| `POST /api/inference/upload` | Inférence sur une image uploadée |
| `GET /api/history` | Historique paginé/triable des runs |
| `GET /api/history/{run_id}` | Détail d'un run (JSON + annotations) |
| `POST /api/history/{run_id}/annotations` | Sauvegarde des annotations utilisateur |
| `DELETE /api/history/{run_id}` | Suppression d'un run + fichiers associés |
| `GET /api/preload/status` | État du worker de pré-chargement |
| `POST /api/preload/start` | Démarre le pré-chargement (thread) |
| `POST /api/preload/stop` | Arrête le pré-chargement |

> Fonctionnalités UI détaillées (édition masque/keypoints, export/import JSON,
> tri par colonne, badges espèce/BCS) : section *Application web* du
> [../README.md](../README.md).

## `db.py` — persistance SQLite (SQLAlchemy)

Base `sqlite:///data/bcs_app.db`. Deux tables.

### `InferenceRun` (`inference_runs`)
Colonnes principales : `id`, `created_at`, `last_inferred_at` (indexée, re-bumpée
à chaque `save_run` idempotent), `image_name`, `source_type`, `dataset`,
`group_name`, `ground_truth`, `image_width/height`, `seg_backend`, `sam2_mode`,
`predicted_class`, `predicted_confidence`, **`predicted_species`**, `num_pose_detections`,
`best_pose_conf`, **`predicted_bcs`**, **`bcs_category`**, `output_path`.

- **Contrainte d'unicité** `uq_run_image_dataset_group_backend`
  (`image_name`, `dataset`, `group_name`, `seg_backend`) : empêche les doublons
  entre le worker de preload et un appel interactif. Les NULL (images uploadées)
  sont traités comme distincts par SQLite → jamais bloqués.
- `to_summary()` : dict résumé pour l'historique (arrondi confiance/BCS).

### `UserAnnotation` (`user_annotations`)
`run_id` (FK unique, cascade delete), `updated_at`, `boxes`, `keypoints`,
`kpt_confs`, `box_confs`, `comments` (JSON en `Text`), `mask_path`.

### Helpers
| Fonction | Rôle |
|---|---|
| `init_db(db_path)` | `create_all` + `_migrate_schema` (idempotent) |
| `session_scope(factory)` | Context manager : rollback on error, close toujours. **Ne commit pas** |
| `save_run(session, ...)` | Persiste un run (idempotent via contrainte d'unicité + bump `last_inferred_at`). Commit lui-même |
| `save_annotations(session, run_id, ...)` | Sauve les annotations utilisateur |
| `load_run(session, run_id)` | Charge un run complet |
| `list_runs(session, limit, offset, sort_by, sort_order)` | Résumé paginé/trié. `sort_by` **validé contre une allowlist** (anti-injection), défaut `last_inferred_at desc` |
| `delete_run(session, run_id)` | Supprime run + fichiers |

⚠️ **Migrations** : `create_all` est un no-op sur les tables existantes. Toute
colonne/contrainte ajoutée après le schéma initial **doit** être backfillée dans
`_migrate_schema` (ex. `predicted_bcs`, `bcs_category`, `predicted_species` y sont
ajoutées via `ALTER TABLE`).

## Frontend — `templates/` + `static/`

| Fichier | Rôle |
|---|---|
| `templates/index.html` | Page unique (Jinja2). Formulaires segmentation SAM2/3 regroupés, badges espèce/BCS |
| `static/js/app.js` | Logique UI : appels API, rendu overlays, édition masque/keypoints, historique, badges (`renderSpecies`, `renderBcs`, `bcsColor`) |
| `static/css/app.css` | Styles (dont `.bcs-badge`, `.species-badge`) |
| `static/images/` | Logo et favicon |

Le frontend a été découpé hors du HTML (JS + CSS séparés). En ajoutant un champ au
résultat d'inférence, penser à répercuter : `format_inference_result` (backend) →
rendu `app.js` → colonne éventuelle dans l'historique + `to_summary` (DB).

## Pré-chargement — `scripts/preload_db.py`

- Lance des inférences sur **toutes** les images des datasets et les persiste,
  pour un historique instantané au démarrage.
- Réutilise `run_core_inference` + `format_inference_result` (mêmes signatures que
  `app.py`) : garder les deux appelants synchronisés.
- Warnings de lint connus (f-strings `\u2501` en Python 3.10) — hors périmètre.
