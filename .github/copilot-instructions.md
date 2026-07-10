# Instructions GitHub Copilot

Ce projet documente ses conventions et son architecture pour les agents.

**Avant chaque action, consulte la documentation :**

1. Lis d'abord [AGENTS.md](../AGENTS.md) (protocole complet + conventions).
2. Puis le hub de documentation [docs/README.md](../docs/README.md) et le fichier
   de référence adapté à ta tâche :
   - Inférence / cascade / package / CLI → [docs/architecture.md](../docs/architecture.md)
   - Entraînement / modèles / datamodules / configs → [docs/training.md](../docs/training.md)
   - App web / base SQLite / frontend → [docs/webapp.md](../docs/webapp.md)
   - Reprise cascade BCS → [docs/BCS_CASCADE_PROGRESS.md](../docs/BCS_CASCADE_PROGRESS.md)

Si la documentation contredit le code, le **code fait foi** : mets la doc à jour
dans le même changement. Respecte les règles listées dans `AGENTS.md` (git,
checkpoints, configs, commits en français).
