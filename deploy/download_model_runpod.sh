#!/bin/bash
# =============================================================================
# USF BIOS - Model Download Script for RunPod
# Downloads arpitsh018/usf-omega-40b-base to /workspace/models/usf_omega
# =============================================================================

set -e

MODEL_REPO="arpitsh018/usf-omega-40b-base"
MODEL_PATH="/workspace/models/usf_omega"

echo ""
echo "============================================================"
echo "  USF BIOS - Model Download Script"
echo "============================================================"
echo "  Model: $MODEL_REPO"
echo "  Target: $MODEL_PATH"
echo "============================================================"
echo ""

# Step 1: Install huggingface_hub if not available
echo "[1/4] Checking huggingface_hub installation..."
pip install -q huggingface_hub hf_transfer
echo "  ✓ huggingface_hub installed"

# Step 2: Login to HuggingFace
echo ""
echo "[2/4] Logging in to HuggingFace..."
echo "  Please enter your HuggingFace token when prompted."
echo "  Get your token from: https://huggingface.co/settings/tokens"
echo ""
huggingface-cli login

# Step 3: Create target directory
echo ""
echo "[3/4] Creating target directory..."
mkdir -p "$MODEL_PATH"
echo "  ✓ Directory created: $MODEL_PATH"

# Step 4: Download model
echo ""
echo "[4/4] Downloading model (this may take a while for 40B model)..."
echo "  Using HF_HUB_ENABLE_HF_TRANSFER for faster downloads..."
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download "$MODEL_REPO" \
    --local-dir "$MODEL_PATH" \
    --local-dir-use-symlinks False

echo ""
echo "============================================================"
echo "  ✓ Model downloaded successfully!"
echo "============================================================"
echo "  Location: $MODEL_PATH"
echo ""
echo "  Verifying model files..."
ls -la "$MODEL_PATH"
echo ""

# Verify config.json exists
if [ -f "$MODEL_PATH/config.json" ]; then
    echo "  ✓ config.json found"
    echo "  Architecture: $(grep -o '"architectures":\s*\["[^"]*"\]' "$MODEL_PATH/config.json" || echo 'Check manually')"
else
    echo "  ⚠ Warning: config.json not found"
fi

echo ""
echo "============================================================"
echo "  Model is ready for USF BIOS training!"
echo "============================================================"
echo ""
