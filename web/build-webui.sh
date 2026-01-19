#!/bin/bash
# USF BIOS Web UI - Build Script
# Builds frontend + backend Docker image without ML dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  USF BIOS Web UI - Docker Build"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "$SCRIPT_DIR/Dockerfile.webui" ]; then
    echo "ERROR: Dockerfile.webui not found in $SCRIPT_DIR"
    exit 1
fi

# Verify required files exist
echo "Verifying required files..."
REQUIRED_FILES=(
    "backend/app/models/db_models.py"
    "backend/app/models/schemas.py"
    "backend/app/models/__init__.py"
    "backend/main.py"
    "backend/app/main.py"
    "frontend/package.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$file" ]; then
        echo "  ✗ Missing: $file"
        exit 1
    fi
    echo "  ✓ $file"
done

echo ""
echo "Building Docker image..."
echo ""

# Build with no cache to ensure fresh build
cd "$SCRIPT_DIR"
docker build \
    --no-cache \
    -f Dockerfile.webui \
    -t usf-bios-webui:latest \
    .

echo ""
echo "=============================================="
echo "  Build Complete!"
echo "=============================================="
echo ""
echo "Run the container:"
echo "  docker run -p 3000:3000 -p 8000:8000 usf-bios-webui:latest"
echo ""
echo "Access:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
