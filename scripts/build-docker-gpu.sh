#!/bin/bash
# ============================================================================
# USF BIOS - GPU Docker Build Script
# Run this on H200/H100 GPU server for fast CUDA compilation
# ============================================================================
#
# USAGE:
#   ./scripts/build-docker-gpu.sh [OPTIONS] [VERSION]
#
# OPTIONS:
#   --no-cache    Disable Docker layer caching (fresh build)
#   --no-push     Build only, do not push to Docker Hub
#   --help        Show this help message
#
# EXAMPLES:
#   ./scripts/build-docker-gpu.sh                    # Build with cache, push
#   ./scripts/build-docker-gpu.sh --no-cache         # Fresh build, push
#   ./scripts/build-docker-gpu.sh --no-push          # Build with cache, no push
#   ./scripts/build-docker-gpu.sh --no-cache --no-push 2.0.12  # Fresh build, specific version, no push
#
# REQUIREMENTS:
#   - NVIDIA GPU (H200, H100, A100, etc.)
#   - Docker with nvidia-container-toolkit
#   - ~50GB disk space
#
# BUILD TIME ESTIMATES:
#   - H200: ~15-20 minutes (first build), ~2-5 min (cached)
#   - H100: ~20-25 minutes (first build), ~2-5 min (cached)
#   - A100: ~25-30 minutes (first build), ~3-5 min (cached)
#   - GitHub Actions (no GPU): 6+ hours (timeout)
#
# CACHING:
#   Default: ON (uses BuildKit cache for fast rebuilds)
#   Use --no-cache to force a fresh build when needed
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
NO_CACHE=false
NO_PUSH=false
VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --no-push)
            NO_PUSH=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [VERSION]"
            echo ""
            echo "OPTIONS:"
            echo "  --no-cache    Disable Docker layer caching (fresh build)"
            echo "  --no-push     Build only, do not push to Docker Hub"
            echo "  --help        Show this help message"
            echo ""
            echo "EXAMPLES:"
            echo "  $0                           # Build with cache, push"
            echo "  $0 --no-cache                # Fresh build, push"
            echo "  $0 --no-push                 # Build with cache, no push"
            echo "  $0 --no-cache --no-push      # Fresh build, no push"
            echo "  $0 2.0.12                    # Specific version"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            VERSION="$1"
            shift
            ;;
    esac
done

# Navigate to project root first (needed for version extraction)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Extract version dynamically from usf_bios/version.py if not provided
if [ -z "$VERSION" ]; then
    DYNAMIC_VERSION=$(python3 -c "exec(open('usf_bios/version.py').read()); print(__version__)" 2>/dev/null || echo "2.0.15")
    VERSION="$DYNAMIC_VERSION"
    echo -e "${GREEN}  Using version from version.py: ${VERSION}${NC}"
else
    # Version was provided - update version.py to match
    echo -e "${YELLOW}  Updating version.py to: ${VERSION}${NC}"
    CURRENT_DATE=$(date '+%Y-%m-%d %H:%M:%S')
    cat > usf_bios/version.py << EOF
# USF BIOS - AI Training & Fine-tuning Platform
# Powered by US Inc
# Make sure to modify __release_datetime__ to release time when making official release.
__version__ = '${VERSION}'
__product_name__ = 'USF BIOS'
__company__ = 'US Inc'
# default release datetime for branches under active development
__release_datetime__ = '${CURRENT_DATE}'
EOF
    echo -e "${GREEN}  ✓ version.py updated to ${VERSION}${NC}"
fi

# Docker Hub image name
IMAGE_NAME="arpitsh018/usf-bios"
DOCKERFILE="web/Dockerfile.gpu"

# Build cache flag
if [ "$NO_CACHE" = true ]; then
    CACHE_FLAG="--no-cache"
    CACHE_STATUS="DISABLED (fresh build)"
else
    CACHE_FLAG=""
    CACHE_STATUS="ENABLED (fast rebuild)"
fi

echo -e "${GREEN}"
echo "============================================================================"
echo "  USF BIOS - GPU Docker Build"
echo "  Version: ${VERSION}"
echo "  Image: ${IMAGE_NAME}:${VERSION}"
echo "  Cache: ${CACHE_STATUS}"
echo "  Push: $([ "$NO_PUSH" = true ] && echo 'DISABLED' || echo 'ENABLED')"
echo "============================================================================"
echo -e "${NC}"

# Check for NVIDIA GPU
echo -e "${YELLOW}[1/5] Checking GPU availability...${NC}"
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. This script requires NVIDIA GPU.${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo -e "${GREEN}  ✓ GPU detected: ${GPU_NAME}${NC}"

# Check Docker
echo -e "${YELLOW}[2/5] Checking Docker...${NC}"
if ! docker info &> /dev/null; then
    echo -e "${RED}ERROR: Docker not running.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Docker is running${NC}"

# Check nvidia-container-toolkit
echo -e "${YELLOW}[3/5] Checking nvidia-container-toolkit...${NC}"
if ! docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-container-toolkit not configured.${NC}"
    echo "Install with: sudo apt-get install -y nvidia-container-toolkit"
    exit 1
fi
echo -e "${GREEN}  ✓ nvidia-container-toolkit working${NC}"

# Already at project root
echo -e "${YELLOW}[4/5] Building from: $(pwd)${NC}"

# Build with GPU support and BuildKit
echo -e "${YELLOW}[5/5] Starting Docker build (BuildKit enabled)...${NC}"
echo ""

# Enable BuildKit for better caching and parallel builds
export DOCKER_BUILDKIT=1

# GitHub token for private usf-transformers repo (REQUIRED)
GITHUB_TOKEN="${2:-${GITHUB_TOKEN:-}}"
if [ -z "${GITHUB_TOKEN}" ]; then
    echo -e "${RED}ERROR: GITHUB_TOKEN required for private usf-transformers repo${NC}"
    echo "Usage: $0 [image-name] [github-token]"
    echo "   Or: export GITHUB_TOKEN=ghp_xxx && $0"
    exit 1
fi
echo -e "${GREEN}✓ GitHub token provided${NC}"

# Use regular docker build (has access to host GPU for compilation)
# Note: buildx with docker-container driver does NOT have GPU access
docker build \
    ${CACHE_FLAG} \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME}:${VERSION}" \
    --tag "${IMAGE_NAME}:latest" \
    --build-arg GITHUB_TOKEN="${GITHUB_TOKEN}" \
    --build-arg DS_BUILD_OPS=1 \
    --build-arg DS_BUILD_AIO=1 \
    --build-arg DS_BUILD_SPARSE_ATTN=1 \
    --build-arg DS_BUILD_TRANSFORMER=1 \
    --build-arg DS_BUILD_TRANSFORMER_INFERENCE=1 \
    --build-arg DS_BUILD_STOCHASTIC_TRANSFORMER=1 \
    --build-arg DS_BUILD_FUSED_ADAM=1 \
    --build-arg DS_BUILD_FUSED_LAMB=1 \
    --build-arg DS_BUILD_CPU_ADAM=1 \
    --build-arg DS_BUILD_CPU_LION=1 \
    --build-arg DS_BUILD_UTILS=1 \
    --build-arg DS_BUILD_EVOFORMER_ATTN=1 \
    --build-arg DS_BUILD_RANDOM_LTD=1 \
    --build-arg DS_BUILD_INFERENCE_CORE_OPS=1 \
    --build-arg DS_BUILD_CUTLASS_OPS=1 \
    --build-arg DS_BUILD_RAGGED_DEVICE_OPS=1 \
    --progress=plain \
    .

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}  ✓ BUILD COMPLETE${NC}"
echo -e "${GREEN}  Image: ${IMAGE_NAME}:${VERSION}${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""

# Show image info
docker images "${IMAGE_NAME}:${VERSION}"

# Push to Docker Hub (unless --no-push specified)
if [ "$NO_PUSH" = true ]; then
    echo ""
    echo -e "${YELLOW}[8/8] Skipping push (--no-push specified)${NC}"
    echo ""
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}  ✓ BUILD COMPLETE (not pushed)${NC}"
    echo -e "${GREEN}  Image: ${IMAGE_NAME}:${VERSION}${NC}"
    echo -e "${GREEN}  Image: ${IMAGE_NAME}:latest${NC}"
    echo -e "${GREEN}============================================================================${NC}"
    echo ""
    echo -e "${YELLOW}To push manually:${NC}"
    echo "  docker push ${IMAGE_NAME}:${VERSION}"
    echo "  docker push ${IMAGE_NAME}:latest"
else
    echo ""
    echo -e "${YELLOW}[8/8] Pushing to Docker Hub...${NC}"
    echo ""

    # Push to Docker Hub
    docker push ${IMAGE_NAME}:${VERSION}
    docker push ${IMAGE_NAME}:latest

    echo ""
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}  ✓ PUSH COMPLETE${NC}"
    echo -e "${GREEN}  Image: ${IMAGE_NAME}:${VERSION}${NC}"
    echo -e "${GREEN}  Image: ${IMAGE_NAME}:latest${NC}"
    echo -e "${GREEN}============================================================================${NC}"
fi
echo ""
echo -e "${YELLOW}To run:${NC}"
echo "  docker run --gpus all -p 3000:3000 ${IMAGE_NAME}:${VERSION}"
