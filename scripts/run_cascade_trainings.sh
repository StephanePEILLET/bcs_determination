#!/usr/bin/env bash
# =============================================================================
# Entraîne, dans l'ordre, les 4 modèles de la cascade BCS puis publie leurs
# checkpoints aux emplacements attendus par l'application / la CLI :
#
#   1. species    → classifieur binaire chien/chat
#   2. dog_breed  → races chien (Stanford, 120 classes)
#   3. cat_breed  → races chat (Oxford, 12 classes)
#   4. bcs_cat    → régression BCS chat (LOCO-CV sur Cats_OGR_dataset)
#
# Le modèle BCS chien reste un placeholder (aucune donnée chien à ce jour).
#
# Usage :
#   bash scripts/run_cascade_trainings.sh              # tout
#   bash scripts/run_cascade_trainings.sh species      # une ou plusieurs étapes
#   bash scripts/run_cascade_trainings.sh dog_breed cat_breed
#
# Variables d'environnement optionnelles :
#   CONDA_ENV=bcs_analysis   # nom de l'environnement conda à activer
#   SKIP_CONDA=1             # ne pas tenter d'activer conda (déjà actif)
# =============================================================================
set -euo pipefail

# ── Localisation : ce script vit dans <repo>/scripts/ ────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-bcs_analysis}"
LOG_DIR="${REPO_ROOT}/experiments/_cascade_logs"
mkdir -p "${LOG_DIR}"

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { printf '\n\033[1;36m[%(%H:%M:%S)T] %s\033[0m\n' -1 "$*"; }
ok()   { printf '\033[1;32m  ✔ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ⚠ %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m  ✘ %s\033[0m\n' "$*" >&2; exit 1; }

# Copie le `last.ckpt` (ou le .ckpt le plus récent) d'une expérience Hydra vers
# le dossier publié attendu par app_checkpoints.py.
publish_ckpt() {
  local experiment_name="$1" dest_dir="$2"
  local src_dir="experiments/${experiment_name}/checkpoints"
  local src=""
  if [[ -f "${src_dir}/last.ckpt" ]]; then
    src="${src_dir}/last.ckpt"
  else
    src="$(ls -t "${src_dir}"/*.ckpt 2>/dev/null | head -n1 || true)"
  fi
  [[ -n "${src}" && -f "${src}" ]] || die "Aucun checkpoint trouvé dans ${src_dir}/"
  mkdir -p "${dest_dir}"
  cp -f "${src}" "${dest_dir}/last.ckpt"
  ok "Publié ${src} → ${dest_dir}/last.ckpt"
}

run_step() {
  local name="$1"; shift
  local logfile="${LOG_DIR}/${name}.log"
  log "▶ Étape « ${name} » — log : ${logfile}"
  # shellcheck disable=SC2068
  if "$@" 2>&1 | tee "${logfile}"; then
    ok "Étape « ${name} » terminée."
  else
    die "Étape « ${name} » a échoué (voir ${logfile})."
  fi
}

# Barrière de vérification : confirme, via app_checkpoints, que le checkpoint
# de l'étape est présent ET résolvable AVANT d'autoriser l'étape suivante.
# Stoppe la cascade (exit) si la sortie attendue n'est pas là.
verify_step() {
  local name="$1"
  log "🔎 Vérification de la sortie de « ${name} » avant de continuer…"
  if python - "${name}" <<'PY'
import sys
from bcs_pipeline.app_checkpoints import (
    resolve_species_ckpt, resolve_dog_breed_ckpt, resolve_cat_breed_ckpt,
    bcs_available,
)

name = sys.argv[1]
checks = {
    "species":   lambda: resolve_species_ckpt(),
    "dog_breed": lambda: resolve_dog_breed_ckpt(),
    "cat_breed": lambda: resolve_cat_breed_ckpt(),
    "bcs_cat":   lambda: ("cat" if bcs_available("cat") else None),
}
resolver = checks.get(name)
if resolver is None:
    print(f"étape inconnue: {name}")
    sys.exit(2)

result = resolver()
if not result:
    print(f"AUCUN checkpoint résolvable pour « {name} »")
    sys.exit(1)
print(f"checkpoint résolu pour « {name} » → {result}")
sys.exit(0)
PY
  then
    ok "Sortie de « ${name} » vérifiée — passage à l'étape suivante autorisé."
  else
    die "Vérification de « ${name} » ÉCHOUÉE : checkpoint absent/non résolvable. Cascade interrompue."
  fi
}

# ── Activation conda (sauf SKIP_CONDA=1) ─────────────────────────────────────
if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    ok "Environnement conda actif : ${CONDA_ENV}"
  else
    warn "conda introuvable — exécution avec le Python courant ($(command -v python))."
  fi
fi

# ── Sélection des étapes ─────────────────────────────────────────────────────
ALL_STEPS=(species dog_breed cat_breed bcs_cat)
if [[ $# -gt 0 ]]; then
  STEPS=("$@")
else
  STEPS=("${ALL_STEPS[@]}")
fi

log "Cascade BCS — étapes : ${STEPS[*]}"

# Affiche une barre de progression GLOBALE de la cascade (étapes terminées / total).
cascade_progress() {
  local done="$1" total="$2" current="$3"
  local width=30
  local filled=$(( width * done / total ))
  local empty=$(( width - filled ))
  local bar=""
  local i
  for (( i = 0; i < filled; i++ )); do bar+="█"; done
  for (( i = 0; i < empty;  i++ )); do bar+="░"; done
  printf '\n\033[1;35m┌─ Cascade %d/%d  [%s]  %d%%  → « %s »\033[0m\n' \
    "$(( done + 1 ))" "${total}" "${bar}" "$(( 100 * done / total ))" "${current}"
}

# ── Exécution ────────────────────────────────────────────────────────────────
TOTAL_STEPS="${#STEPS[@]}"
STEP_IDX=0
for step in "${STEPS[@]}"; do
  cascade_progress "${STEP_IDX}" "${TOTAL_STEPS}" "${step}"
  case "${step}" in
    species)
      run_step species python train.py --config-name config_species
      publish_ckpt vit_species checkpoints/classification/species
      verify_step species
      ;;
    dog_breed)
      run_step dog_breed python train.py --config-name config_dog_breed
      publish_ckpt vit_dog_breed checkpoints/classification/dog_breed
      verify_step dog_breed
      ;;
    cat_breed)
      run_step cat_breed python train.py --config-name config_cat_breed
      publish_ckpt vit_cat_breed checkpoints/classification/cat_breed
      verify_step cat_breed
      ;;
    bcs_cat)
      # Écrit directement dans checkpoints/bcs_regression/cat/fold_*/ — pas de copie.
      run_step bcs_cat python scripts/train_bcs_regression.py --species cat
      verify_step bcs_cat
      ;;
    *)
      die "Étape inconnue : « ${step} » (valeurs : ${ALL_STEPS[*]})"
      ;;
  esac
  STEP_IDX=$(( STEP_IDX + 1 ))
  printf '\033[1;35m└─ Cascade %d/%d terminée(s).\033[0m\n' "${STEP_IDX}" "${TOTAL_STEPS}"
done

# ── Vérification finale de disponibilité ─────────────────────────────────────
log "Vérification des checkpoints publiés"
python - <<'PY'
from bcs_pipeline.app_checkpoints import (
    species_available, dog_breed_available, cat_breed_available, bcs_available,
    resolve_species_ckpt, resolve_dog_breed_ckpt, resolve_cat_breed_ckpt,
)
rows = [
    ("species",   species_available(),   resolve_species_ckpt()),
    ("dog_breed", dog_breed_available(),  resolve_dog_breed_ckpt()),
    ("cat_breed", cat_breed_available(),  resolve_cat_breed_ckpt()),
    ("bcs_cat",   bcs_available("cat"),   None),
    ("bcs_dog",   bcs_available("dog"),   None),
]
for name, avail, path in rows:
    mark = "OK " if avail else "-- "
    extra = f" → {path}" if path else ""
    print(f"  [{mark}] {name:9s} disponible={avail}{extra}")
PY

log "Terminé. La cascade utilisera automatiquement les modèles présents."
