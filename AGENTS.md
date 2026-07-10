# AGENTS.md — Instructions pour agents

> Ce dépôt fournit une **documentation de référence structurée dans `docs/`**.
> **Avant CHAQUE action** (lecture ciblée, écriture, refactor, exécution,
> entraînement), commence par consulter la doc pertinente. Ne devine pas
> l'architecture : elle est décrite.

## Protocole obligatoire « lire avant d'agir »

1. Ouvre **[docs/README.md](docs/README.md)** (le hub) pour identifier le domaine
   concerné et lire les **conventions transverses**.
2. Ouvre le fichier de référence adapté :
   - Inférence, cascade, structure du package, CLI → [docs/architecture.md](docs/architecture.md)
   - Entraînement, LightningModules, datamodules, modèles, configs → [docs/training.md](docs/training.md)
   - App web, base SQLite, frontend, pré-chargement → [docs/webapp.md](docs/webapp.md)
   - Reprise du chantier cascade BCS → [docs/BCS_CASCADE_PROGRESS.md](docs/BCS_CASCADE_PROGRESS.md)
3. Applique les conventions ; si la doc contredit le code, **le code fait foi** et
   tu mets la doc à jour dans le même changement.

## Résumé du projet

Estimation automatique du **Body Condition Score** (BCS, 1–9) d'un animal à partir
d'une image, via une **cascade** : espèce (chien/chat) → race (routée) →
segmentation (silhouette) → BCS (routé par espèce). Stack : PyTorch Lightning +
Hydra, FastAPI (app web), SQLite. Package principal : `src/bcs_pipeline/`.

## Commandes clés

```bash
uv run python app.py                 # app web (http://localhost:8000)
uv run python inference.py --mode full --image_path data/Reddit_example/dog.jpg
python train.py --config-name config_species   # entraînement Hydra
```
Environnements : `uv` (`.venv/`, Py 3.12) ou conda `bcs_analysis` (Py 3.10,
kernel des notebooks).

## Règles à respecter (voir docs/README.md pour le détail)

- **Ne jamais committer** poids/données : `checkpoints/`, `data/`, `experiments/`,
  `*.ckpt/*.pt/*.pth/*.onnx` sont dans `.gitignore` (28 Go locaux).
- Chemins de checkpoints **uniquement** via `src/bcs_pipeline/app_checkpoints.py`.
- Nouvelle valeur `dataset`/`model_name` → l'ajouter à
  `utils/config_validation.py`.
- `run_core_inference` / `format_inference_result` sont partagés `app.py` ↔
  `scripts/preload_db.py` : synchroniser les deux appelants.
- Toute colonne DB ajoutée → backfill dans `db.py::_migrate_schema`.
- Commits : Conventional Commits **en français** (`feat:`, `fix:`, `docs:`…).
  Pas de `--no-verify`/`--force`/`reset --hard` ; demander avant `git push`.
- Ne pas exécuter SAM 3 / entraînement sans GPU sauf demande explicite.
- ⚠️ Le motif `.gitignore` `data/` masque `src/bcs_pipeline/data/` aux outils de
  recherche : lire ces datamodules directement.

## Maintenance de la doc

Après un changement structurel, mets à jour le fichier `docs/` concerné **dans le
même commit**.
