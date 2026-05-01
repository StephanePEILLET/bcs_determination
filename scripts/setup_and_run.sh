#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Body Pawsitive — Setup & Run (100% non-interactif)
#
# Installe automatiquement uv si absent, crée l'environnement, détecte le GPU
# (CUDA / MPS / CPU), télécharge les données et lance l'application.
# Ne demande JAMAIS d'input utilisateur.
#
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh              # tout faire
#   ./setup_and_run.sh --skip-data  # passer le téléchargement des données
#   ./setup_and_run.sh --skip-env   # passer l'installation de l'environnement
#   ./setup_and_run.sh --no-launch  # ne pas lancer l'app à la fin
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Couleurs ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[  OK]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[FAIL]${NC}  $*"; }
section() { echo -e "\n${CYAN}${BOLD}━━━ $* ━━━${NC}\n"; }

# ── Configuration ────────────────────────────────────────────────────────────
# Navigate to the project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="3.10"

# URLs des données
STANFORD_URL="http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
OXFORD_IMAGES_URL="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
OXFORD_ANNOT_URL="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
SAM2_CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
REDDIT_IMG1_URL="https://preview.redd.it/is-my-dog-overweight-v0-t6vyhitvecng1.jpg?width=1080&crop=smart&auto=webp&s=50ea739e28b4918e60c1c690f7301b69d3b89c4b"
REDDIT_IMG2_URL="https://preview.redd.it/is-my-dog-overweight-v0-nvhuvntvecng1.jpg?width=1080&crop=smart&auto=webp&s=49987ac045a56da4b81855f848784fa296f3db27"

# Chemins locaux
DATA_DIR="$SCRIPT_DIR/data"
STANFORD_DIR="$DATA_DIR/stanford_dogs/images"
OXFORD_DIR="$DATA_DIR/Oxford-IIIT_pet_dataset"
REDDIT_DIR="$DATA_DIR/Reddit_example"
SAM2_CKPT="$SCRIPT_DIR/checkpoints/segmentation/sam2.1_hiera_large.pt"

# GPU détecté (set in detect_gpu)
GPU_TYPE="cpu"

# ── Arguments CLI ────────────────────────────────────────────────────────────
SKIP_DATA=false
SKIP_ENV=false
NO_LAUNCH=false
FORCE_CPU=false
PRELOAD_DB=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-data)  SKIP_DATA=true; shift ;;
        --skip-env)   SKIP_ENV=true; shift ;;
        --no-launch)  NO_LAUNCH=true; shift ;;
        --cpu)        FORCE_CPU=true; shift ;;
        --preload)    PRELOAD_DB=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--skip-data] [--skip-env] [--no-launch] [--cpu] [--preload]"
            echo ""
            echo "Options:"
            echo "  --skip-data   Passer le téléchargement des données"
            echo "  --skip-env    Passer l'installation de l'environnement"
            echo "  --no-launch   Ne pas lancer l'app à la fin"
            echo "  --cpu         Forcer l'exécution sur CPU (ignore GPU)"
            echo "  --preload     Pré-charger la base de données (inférences sur toutes les images)"
            exit 0 ;;
        *)
            error "Argument inconnu : $1"; exit 1 ;;
    esac
done

# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 0 : Détection du GPU
# ═════════════════════════════════════════════════════════════════════════════
detect_gpu() {
    section "Détection du matériel"

    # --cpu force le mode CPU
    if [[ "$FORCE_CPU" == true ]]; then
        GPU_TYPE="cpu"
        export BCS_DEVICE="cpu"
        info "Mode CPU forcé via ${BOLD}--cpu${NC}"
        warn "Les modèles tourneront sur CPU (plus lent mais compatible partout)"
        return
    fi

    # 1. Vérifier NVIDIA CUDA
    if command -v nvidia-smi &>/dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        if [[ -n "$gpu_name" ]]; then
            GPU_TYPE="cuda"
            export BCS_DEVICE="cuda"
            success "GPU NVIDIA détecté : ${BOLD}$gpu_name${NC}"

            local cuda_version
            cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
            info "Driver NVIDIA : $cuda_version"
            return
        fi
    fi

    # 2. Vérifier Apple Silicon (MPS)
    if [[ "$(uname -s)" == "Darwin" ]]; then
        local chip
        chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
        if [[ "$chip" == *"Apple"* ]]; then
            GPU_TYPE="mps"
            export BCS_DEVICE="mps"
            success "GPU Apple Silicon détecté : ${BOLD}$chip${NC}"
            info "Le backend MPS (Metal Performance Shaders) sera utilisé"
            return
        fi
    fi

    # 3. Fallback CPU — aucun GPU détecté
    GPU_TYPE="cpu"
    export BCS_DEVICE="cpu"
    warn "Aucun GPU détecté — l'application tournera sur CPU"
    info "Utilisez ${BOLD}--cpu${NC} pour supprimer cet avertissement"
}

# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : Auto-installation de uv
# ═════════════════════════════════════════════════════════════════════════════
ensure_uv() {
    # Vérifier dans le PATH standard + ~/.local/bin + ~/.cargo/bin
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if command -v uv &>/dev/null; then
        success "uv déjà installé ($(uv --version 2>/dev/null || echo '?'))"
        return 0
    fi

    section "Installation automatique de uv"

    if command -v curl &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1
    elif command -v wget &>/dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh 2>&1
    else
        error "Ni curl ni wget disponible pour installer uv."
        exit 1
    fi

    # Recharger le PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if command -v uv &>/dev/null; then
        success "uv installé ($(uv --version 2>/dev/null || echo '?'))"
    else
        error "Échec de l'installation de uv"
        exit 1
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : Environnement Python
# ═════════════════════════════════════════════════════════════════════════════
setup_env() {
    if [[ "$SKIP_ENV" == true ]]; then
        warn "Étape environnement ignorée (--skip-env)"
        if [[ -f ".venv/bin/activate" ]]; then
            # shellcheck disable=SC1091
            source .venv/bin/activate
        fi
        return 0
    fi

    section "Configuration de l'environnement Python"

    # Crée le venv
    if [[ ! -d ".venv" ]]; then
        info "Création du virtualenv Python $PYTHON_VERSION ..."
        uv venv --python "$PYTHON_VERSION" .venv
        success "Virtualenv créé (.venv/)"
    else
        success "Virtualenv .venv/ existe déjà"
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate

    # ── Installation PyTorch avec le bon backend GPU ─────────────────────────
    info "Installation de PyTorch (backend: ${BOLD}$GPU_TYPE${NC}) ..."

    case "$GPU_TYPE" in
        cuda)
            # Détecte la version CUDA installée pour choisir le bon index
            local cuda_major
            cuda_major=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' | head -1 || echo "11")
            if [[ "$cuda_major" -ge 12 ]]; then
                info "CUDA ≥ 12 détecté → cu124"
                uv pip install --quiet \
                    torch torchvision torchaudio \
                    --index-url https://download.pytorch.org/whl/cu124
            else
                info "CUDA < 12 → cu118"
                uv pip install --quiet \
                    torch torchvision torchaudio \
                    --index-url https://download.pytorch.org/whl/cu118
            fi
            ;;
        mps|cpu)
            # Sur macOS (MPS) et CPU, le build par défaut de PyPI suffit
            # MPS est automatiquement disponible avec torch >= 2.0 sur Apple Silicon
            uv pip install --quiet torch torchvision torchaudio
            ;;
    esac
    success "PyTorch installé"

    # ── Installation du projet ───────────────────────────────────────────────
    info "Installation du projet (pyproject.toml + extras dev) ..."
    uv pip install --quiet -e ".[dev]"
    success "Projet bcs-pipeline installé"

    # ── Dépendances supplémentaires ──────────────────────────────────────────
    # Packages présents dans environment.yaml ou utilisés dans le code
    # mais absents du pyproject.toml
    info "Installation des dépendances supplémentaires ..."
    uv pip install --quiet \
        python-multipart \
        scikit-learn \
        seaborn \
        pandas \
        tensorboard \
        flask \
        hydra-optuna-sweeper \
        sqlalchemy \
        jinja2 \
        sam2 \
        scipy \
        opencv-python-headless
    success "Dépendances supplémentaires installées"

    # ── Vérification des imports + device GPU ────────────────────────────────
    info "Vérification des imports et du device ..."
    python -c "
import torch
import pytorch_lightning
import fastapi
import sqlalchemy

# Détection du device
if torch.cuda.is_available():
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    print(f'  🟢 GPU CUDA : {gpu_name}')
    print(f'  CUDA {torch.version.cuda}  |  cuDNN {torch.backends.cudnn.version()}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    print(f'  🟢 GPU MPS (Apple Silicon) activé')
else:
    device = 'cpu'
    print(f'  🟡 CPU uniquement')

print(f'  PyTorch {torch.__version__}  |  Device: {device}')
print(f'  Lightning {pytorch_lightning.__version__}')
print(f'  FastAPI {fastapi.__version__}')
print(f'  SQLAlchemy {sqlalchemy.__version__}')
" || { error "Vérification des imports échouée"; exit 1; }

    success "Environnement prêt"
}

# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : Téléchargement des données
# ═════════════════════════════════════════════════════════════════════════════

download() {
    local url="$1"
    local dest="$2"
    local name
    name="$(basename "$dest")"

    if [[ -f "$dest" ]]; then
        success "$name déjà présent"
        return 0
    fi

    info "Téléchargement de $name ..."
    mkdir -p "$(dirname "$dest")"

    if command -v curl &>/dev/null; then
        curl -fL --progress-bar --retry 3 --retry-delay 5 -o "$dest" "$url"
    elif command -v wget &>/dev/null; then
        wget --no-verbose --tries=3 --wait=5 -O "$dest" "$url"
    else
        error "Ni curl ni wget disponible"
        exit 1
    fi
    success "$name téléchargé"
}

download_data() {
    if [[ "$SKIP_DATA" == true ]]; then
        warn "Étape données ignorée (--skip-data)"
        return 0
    fi

    section "Téléchargement des données et checkpoints"
    mkdir -p "$DATA_DIR"

    # ── Stanford Dogs ────────────────────────────────────────────────────────
    if [[ -d "$STANFORD_DIR/Images" ]] && [[ -n "$(ls -A "$STANFORD_DIR/Images/" 2>/dev/null)" ]]; then
        success "Stanford Dogs déjà présent"
    else
        local stanford_tar="$DATA_DIR/stanford_images.tar"
        download "$STANFORD_URL" "$stanford_tar"
        info "Extraction de Stanford Dogs ..."
        mkdir -p "$STANFORD_DIR"
        tar -xf "$stanford_tar" -C "$STANFORD_DIR"
        rm -f "$stanford_tar"
        success "Stanford Dogs extrait"
    fi

    # ── Oxford-IIIT Pet ──────────────────────────────────────────────────────
    if [[ -d "$OXFORD_DIR/images" ]] && [[ -n "$(ls -A "$OXFORD_DIR/images/" 2>/dev/null)" ]]; then
        success "Oxford-IIIT Pet déjà présent"
    else
        local oxford_img="$DATA_DIR/oxford_images.tar.gz"
        local oxford_ann="$DATA_DIR/oxford_annotations.tar.gz"
        download "$OXFORD_IMAGES_URL" "$oxford_img"
        download "$OXFORD_ANNOT_URL" "$oxford_ann"
        info "Extraction d'Oxford-IIIT Pet ..."
        mkdir -p "$OXFORD_DIR"
        tar -xzf "$oxford_img" -C "$OXFORD_DIR"
        tar -xzf "$oxford_ann" -C "$OXFORD_DIR"
        rm -f "$oxford_img" "$oxford_ann"
        success "Oxford-IIIT Pet extrait"
    fi

    # ── Reddit (images d'exemple) ────────────────────────────────────────────
    mkdir -p "$REDDIT_DIR"
    download "$REDDIT_IMG1_URL" "$REDDIT_DIR/reddit_dog_1.jpg"
    download "$REDDIT_IMG2_URL" "$REDDIT_DIR/reddit_dog_2.jpg"

    # ── SAM 2.1 checkpoint ───────────────────────────────────────────────────
    download "$SAM2_CKPT_URL" "$SAM2_CKPT"

    # ── Checkpoints entraînés ────────────────────────────────────────────────
    echo ""
    # Paths matching app.py constants
    local cls_ckpt="checkpoints/classification/vit_dogs_cats/last.ckpt"
    local seg_ckpt="checkpoints/segmentation/deeplabv3_resnet50_last-v1.ckpt"
    local pose_ckpt="checkpoints/pose/yolo_best.pt"

    [[ -f "$cls_ckpt" ]]  && success "Classification checkpoint OK"  || warn "Classification checkpoint manquant (entraînement requis)"
    [[ -f "$seg_ckpt" ]]  && success "Segmentation checkpoint OK"    || warn "Segmentation checkpoint manquant (entraînement requis)"
    [[ -f "$pose_ckpt" ]] && success "Pose checkpoint OK"             || warn "Pose checkpoint manquant (entraînement requis)"
}

# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 : Pré-chargement de la base de données
# ═════════════════════════════════════════════════════════════════════════════
preload_database() {
    if [[ "$PRELOAD_DB" != true ]]; then
        return 0
    fi

    if [[ "$SKIP_DATA" == true ]]; then
        return 0
    fi

    local cls_ckpt="checkpoints/classification/vit_dogs_cats/last.ckpt"
    if [[ ! -f "$cls_ckpt" ]]; then
        warn "Checkpoints d'entraînement manquants — pré-chargement ignoré"
        info "Les inférences seront calculées à la demande via l'interface web"
        return 0
    fi

    section "Pré-chargement de la base de données"
    info "Lancement du pré-chargement des inférences ..."
    python scripts/preload_db.py
    success "Base de données pré-chargée"
}

# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 : Lancement
# ═════════════════════════════════════════════════════════════════════════════
launch_app() {
    if [[ "$NO_LAUNCH" == true ]]; then
        warn "Lancement ignoré (--no-launch)"
        success "Setup terminé. Lancez : source .venv/bin/activate && python app.py"
        return 0
    fi

    section "Lancement de Body Pawsitive"
    info "Device : ${BOLD}$GPU_TYPE${NC}"
    info "URL    : ${BOLD}http://localhost:5000${NC}"
    info "Ctrl+C pour arrêter.\n"
    python app.py
}

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
main() {
    echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║          🐾  Body Pawsitive — Setup & Run           ║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}\n"

    detect_gpu
    ensure_uv
    setup_env
    download_data
    preload_database
    launch_app
}

main
