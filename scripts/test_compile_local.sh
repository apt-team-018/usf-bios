#!/bin/bash
# Local test script for Cython compilation
# Tests the compile process without waiting for Docker (30 min -> 2 min)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="/tmp/usf_compile_test"

echo "=============================================="
echo "  Local Cython Compilation Test"
echo "=============================================="

# Clean previous test
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv "$TEST_DIR/venv"
source "$TEST_DIR/venv/bin/activate"

# Install Cython and setuptools
echo "[2/5] Installing Cython and setuptools..."
pip install --quiet cython setuptools

# Copy source files (simulating Docker COPY)
echo "[3/5] Copying source files..."
mkdir -p "$TEST_DIR/compile"
cp -r "$PROJECT_DIR/usf_bios" "$TEST_DIR/compile/"
cp -r "$PROJECT_DIR/web/backend" "$TEST_DIR/compile/web_backend"
cp "$PROJECT_DIR/scripts/compile_to_so.py" "$TEST_DIR/compile/"

# Modify compile script paths for local test
cd "$TEST_DIR/compile"

# Create a modified version of the compile script for local testing
cat > test_compile.py << 'EOF'
import sys
sys.path.insert(0, '.')
exec(open('compile_to_so.py').read().replace(
    "'/compile/usf_bios'", "'./usf_bios'"
).replace(
    "'/compile/web/backend'", "'./web_backend'"
).replace(
    "'/compile'", "'.'"
).replace(
    "compile_from='/compile'", "compile_from='.'"
).replace(
    "compile_from='/compile/web/backend'", "compile_from='./web_backend'"
))
EOF

# Run compilation
echo "[4/5] Running Cython compilation..."
python test_compile.py

# Test import of web backend
echo "[5/5] Testing imports..."
cd web_backend
python -c "
import sys
sys.path.insert(0, '.')
try:
    from app.core.config import settings
    print('  ✓ config.py imports correctly')
except Exception as e:
    print(f'  ✗ config.py failed: {e}')
    sys.exit(1)

try:
    from app.models import TrainingConfig
    print('  ✓ models/__init__.py imports correctly')
except Exception as e:
    print(f'  ✗ models/__init__.py failed: {e}')
    sys.exit(1)
"

echo ""
echo "=============================================="
echo "  ✓ LOCAL TEST PASSED"
echo "  Safe to push to GitHub"
echo "=============================================="

deactivate
