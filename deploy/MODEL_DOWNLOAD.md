# Model Download Guide for RunPod

## Prerequisites
- RunPod instance with GPU
- HuggingFace account with access to the model

## Steps

### 1. Install HuggingFace Hub
```bash
pip install huggingface_hub hf_transfer
```

### 2. Login to HuggingFace
```bash
huggingface-cli login
```
Enter your token from: https://huggingface.co/settings/tokens

### 3. Download Model
```bash
# Create target directory
mkdir -p /workspace/models/usf_omega

# Download with fast transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download arpitsh018/usf-omega-40b-base \
    --local-dir /workspace/models/usf_omega
```

## Quick One-Liner
```bash
pip install huggingface_hub hf_transfer && huggingface-cli login && mkdir -p /workspace/models/usf_omega && HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download arpitsh018/usf-omega-40b-base --local-dir /workspace/models/usf_omega --local-dir-use-symlinks False
```

## Verify Download
```bash
ls -la /workspace/models/usf_omega
cat /workspace/models/usf_omega/config.json | grep architectures
```
