# RLHF KTO - Binary Feedback Format

> **For USF-BIOS Frontend UI Training**

---

## Quick Start

Upload a `.jsonl` file with conversations and a `label` (true/false).

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help?"}], "label": true}
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "What?"}], "label": false}
```

---

## How to Use in USF-BIOS UI

### Step 1: Select Model
Choose your base model (should be SFT-trained first).

### Step 2: Configure Dataset
- Click **"Upload"** tab
- Upload your `.jsonl` file
- System auto-detects as **"KTO Binary Feedback"** format

### Step 3: Training Settings
| Setting | Value |
|---------|-------|
| **Training Method** | `RLHF` (auto-selected) |
| **RLHF Algorithm** | `KTO` (required for this format) |
| **Train Type** | `LoRA` / `QLoRA` / `AdaLoRA` / `Full` |
| **Beta** | 0.1 (default) |
| **Epochs** | 1-2 |

### Step 4: Review & Start
Click **"Start Training"** to begin.

---

## Format Structure

| Field | Required | Description |
|-------|----------|-------------|
| `messages` | ✅ Yes | Conversation array |
| `label` | ✅ Yes | `true` = good, `false` = bad |

Alternative format (also supported):
| Field | Required | Description |
|-------|----------|-------------|
| `prompt` | ✅ Yes | User question |
| `completion` | ✅ Yes | Model response |
| `label` | ✅ Yes | `true` = good, `false` = bad |

---

## Examples

### Good Response (label: true)
```json
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is artificial intelligence - machines that simulate human cognitive functions like learning and problem-solving."}], "label": true}
```

### Bad Response (label: false)
```json
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "I don't know."}], "label": false}
```

### With System Prompt
```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I assist you today?"}], "label": true}
```

---

## When to Use KTO vs DPO

| Scenario | Use |
|----------|-----|
| Thumbs up/down data | **KTO** ✅ |
| Production user feedback | **KTO** ✅ |
| Comparative preferences | DPO |
| Paired chosen/rejected | DPO |

**KTO is easier to collect data for** - you just need good/bad labels, not pairs.

---

## Best Practices

1. **Balance Labels** - Aim for ~50% true, ~50% false
2. **Clear Quality** - Good examples should be clearly good
3. **SFT First** - Always do SFT training before KTO
4. **More Data** - KTO needs ~2x more samples than DPO

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Format not detected | Ensure `messages` and `label` fields exist |
| Imbalanced labels | Balance to 40-60% positive |
| No improvement | Increase dataset size, check label quality |
| Forgetting | Lower learning rate, fewer epochs |
