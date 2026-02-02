# RLHF Online - Prompts Format

> **For USF-BIOS Frontend UI Training**

---

## Quick Start

Upload a `.jsonl` file with prompts only (no responses). The model generates responses during training.

```json
{"prompt": "Write a professional email declining a meeting."}
{"prompt": "Explain quantum computing simply."}
```

Or messages format:
```json
{"messages": [{"role": "user", "content": "How do I learn Python?"}]}
```

---

## How to Use in USF-BIOS UI

### Step 1: Select Model
Choose your SFT-trained model as base. **You also need a reward model.**

### Step 2: Configure Dataset
- Click **"Upload"** tab
- Upload your prompts `.jsonl` file
- System auto-detects as **"RLHF Online"** format

### Step 3: Training Settings
| Setting | Value |
|---------|-------|
| **Training Method** | `RLHF` (auto-selected) |
| **RLHF Algorithm** | `PPO` / `GRPO` / `GKD` |
| **Reward Model** | Select your trained RM |
| **Train Type** | `LoRA` / `QLoRA` / `AdaLoRA` / `Full` |
| **KL Coefficient** | 0.05-0.1 |

### Step 4: Review & Start
Click **"Start Training"** to begin.

---

## Format Structure

| Field | Required | Description |
|-------|----------|-------------|
| `prompt` | ✅ Yes* | User prompt string |
| `messages` | ✅ Yes* | Messages array (alternative to prompt) |

*Use either `prompt` OR `messages`, not both.

---

## Supported Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **PPO** | Proximal Policy Optimization | General online RL |
| **GRPO** | Group Relative Policy Optimization | More stable training |
| **GKD** | Generalized Knowledge Distillation | Teacher distillation |

---

## Examples

### Simple Prompts
```json
{"prompt": "Write a haiku about coding"}
{"prompt": "Explain machine learning to a child"}
{"prompt": "What are best practices for API security?"}
```

### With System Prompt
```json
{"messages": [{"role": "system", "content": "You are a coding expert."}, {"role": "user", "content": "Write a Python sort function"}]}
```

### Task-Specific
```json
{"prompt": "Solve step by step: 25 * 4 + 10 = ?"}
{"prompt": "Summarize the benefits of exercise in 3 points"}
```

---

## Requirements

**Before Online RLHF, you need:**

1. **SFT-trained model** - Base model fine-tuned on instructions
2. **Reward model** - To score generated responses
3. **Compute resources** - Online RL needs ~4x more compute than offline

**Training pipeline:**
```
SFT Model → Online RLHF → Aligned Model
              ↑
        Reward Model (scores responses)
```

---

## Online vs Offline RLHF

| Aspect | Online (PPO/GRPO) | Offline (DPO/KTO) |
|--------|-------------------|-------------------|
| Data needed | Prompts only | Preference pairs |
| Compute cost | High (~4x) | Lower |
| Stability | Challenging | Stable |
| Exploration | Yes | No |
| Reward model | **Required** | Not needed |

**Use Offline (DPO) first** - it's simpler and often sufficient. Use Online only if offline plateaus.

---

## Best Practices

1. **SFT First** - Always start with SFT training
2. **Good Reward Model** - Train a quality RM on preference data
3. **Monitor KL** - Keep KL divergence controlled
4. **Diverse Prompts** - Use varied prompts to prevent mode collapse
5. **Start Small** - Test with fewer prompts first

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Format not detected | Ensure `prompt` or `messages` field exists |
| Reward hacking | Add KL penalty, diversify prompts |
| Training unstable | Reduce learning rate, increase KL coefficient |
| Mode collapse | Add temperature, increase exploration |
| No improvement | Check reward model quality |
