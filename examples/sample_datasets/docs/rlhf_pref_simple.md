# RLHF Preference - Simple Format

> **For USF-BIOS Frontend UI Training**

---

## Quick Start

Upload a `.jsonl` file with `prompt`, `chosen`, and `rejected` fields.

```json
{"prompt": "What is AI?", "chosen": "AI is artificial intelligence - machines simulating human intelligence.", "rejected": "I don't know."}
```

---

## How to Use in USF-BIOS UI

### Step 1: Select Model
Choose your base model (should be SFT-trained first for best results).

### Step 2: Configure Dataset
- Click **"Upload"** tab
- Upload your `.jsonl` file
- System auto-detects as **"RLHF Preference"** format

### Step 3: Training Settings
| Setting | Value |
|---------|-------|
| **Training Method** | `RLHF` (auto-selected) |
| **RLHF Algorithm** | `DPO` / `ORPO` / `SimPO` / `CPO` / `RM` |
| **Train Type** | `LoRA` / `QLoRA` / `AdaLoRA` / `Full` |
| **Beta** | 0.1 (default for DPO) |
| **Epochs** | 1-3 |
| **Learning Rate** | 5e-7 |

### Step 4: Review & Start
Click **"Start Training"** to begin.

---

## Format Structure

| Field | Required | Description |
|-------|----------|-------------|
| `prompt` | ✅ Yes | User's question or instruction |
| `chosen` | ✅ Yes | Preferred (better) response |
| `rejected` | ✅ Yes | Non-preferred (worse) response |

---

## Supported Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **DPO** | Direct Preference Optimization | General alignment (recommended) |
| **ORPO** | Odds Ratio Preference Optimization | Efficiency |
| **SimPO** | Simple Preference Optimization | Simplicity |
| **CPO** | Contrastive Preference Optimization | Strong contrast |
| **RM** | Reward Model Training | Training reward models |

---

## Examples

### Helpfulness
```json
{"prompt": "How do I learn programming?", "chosen": "Start with Python - it's beginner-friendly. Try free resources like Codecademy or freeCodeCamp. Build small projects to practice.", "rejected": "Just Google it."}
```

### Safety
```json
{"prompt": "How to pick a lock?", "chosen": "I can't help with that as it could enable illegal activity. If you're locked out, contact a licensed locksmith.", "rejected": "You need a tension wrench and pick. Insert the tension wrench..."}
```

### Accuracy
```json
{"prompt": "What's the capital of France?", "chosen": "Paris is the capital of France.", "rejected": "London is the capital of France."}
```

### Quality
```json
{"prompt": "Explain photosynthesis", "chosen": "Photosynthesis is how plants convert sunlight, water, and CO2 into glucose and oxygen. It occurs in chloroplasts using chlorophyll.", "rejected": "Plants make food from sun."}
```

---

## Training Pipeline

**Recommended workflow:**

1. **SFT First** - Train on instructions (Messages format)
2. **RLHF Second** - Align with preferences (this format)

In the UI:
1. Complete SFT training
2. Use SFT output model as base for RLHF
3. Upload preference dataset
4. Select RLHF → DPO (or other algorithm)

---

## Best Practices

1. **Clear Preference** - Chosen must be objectively better
2. **Same Prompt** - Both responses answer the same question
3. **Meaningful Difference** - Not too obvious, not too subtle
4. **Diverse Topics** - Cover various domains
5. **SFT First** - Always do SFT before RLHF

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Format not detected | Ensure `prompt`, `chosen`, `rejected` fields exist |
| Loss not decreasing | Check that chosen ≠ rejected |
| Model outputs worse | Reduce learning rate, increase beta |
| Forgetting skills | Lower learning rate, fewer epochs |
