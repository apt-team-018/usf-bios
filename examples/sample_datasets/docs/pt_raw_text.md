# Pre-Training Raw Text Format

> **For USF-BIOS Frontend UI Training**

---

## Quick Start

Upload a `.jsonl` file where each line contains a `text` field with raw text content.

```json
{"text": "Your domain-specific text content goes here..."}
```

---

## How to Use in USF-BIOS UI

### Step 1: Select Model
Choose your base model from the UI.

### Step 2: Configure Dataset
- Click **"Upload"** tab
- Upload your `.jsonl` file
- System auto-detects as **"Pre-Training"** format

### Step 3: Training Settings
| Setting | Value |
|---------|-------|
| **Training Method** | `PT` (auto-selected) |
| **Train Type** | `LoRA` / `QLoRA` / `AdaLoRA` / `Full` |
| **Epochs** | 1-2 (PT needs fewer) |
| **Learning Rate** | 5e-6 (lower than SFT) |
| **Max Length** | 2048-4096 |

### Step 4: Review & Start
Click **"Start Training"** to begin.

---

## Format Structure

| Field | Required | Description |
|-------|----------|-------------|
| `text` | ✅ Yes | Raw text content for language modeling |

**All text is trained.** The model learns to predict the next token.

---

## Examples

### Domain Knowledge
```json
{"text": "Quantum entanglement is a phenomenon where particles become interconnected such that the quantum state of each cannot be described independently."}
```

### Technical Documentation
```json
{"text": "The API accepts POST requests with JSON payload. Authentication uses Bearer tokens. Rate limiting: 1000 requests/minute."}
```

### Legal Text
```json
{"text": "In contract law, consideration refers to something of value exchanged between parties. Without valid consideration, a contract may be unenforceable."}
```

### Multi-paragraph
```json
{"text": "Machine learning has revolutionized complex problems.\n\nDeep learning uses neural networks with multiple layers to learn hierarchical data representations."}
```

---

## Use Cases

| Use Case | PT Format? |
|----------|-----------|
| Domain adaptation (medical, legal, etc.) | ✅ Yes |
| Knowledge injection | ✅ Yes |
| Language expansion | ✅ Yes |
| Instruction following | ❌ Use SFT |
| Conversational AI | ❌ Use SFT Messages |
| Preference alignment | ❌ Use RLHF |

---

## Training Pipeline

**Recommended workflow:**

1. **Pre-Train (PT)** on domain text → builds knowledge
2. **Fine-Tune (SFT)** on instructions → teaches task completion
3. **Align (RLHF)** on preferences → improves quality

In the UI:
1. Run PT job, wait for completion
2. Check Training History for output model path
3. Start new job using PT output as base model
4. Use SFT dataset for instruction tuning

---

## Best Practices

1. **Clean Text** - Remove HTML, URLs, excessive whitespace
2. **Quality Content** - Use well-written, accurate sources
3. **Sufficient Length** - Minimum 50 tokens per sample
4. **Deduplicate** - Remove duplicate or near-duplicate content
5. **Lower LR** - Use 5e-6 or lower to prevent forgetting

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Format not detected | Ensure `text` field exists |
| Text too short | Use samples with 50+ tokens |
| Catastrophic forgetting | Lower learning rate, fewer epochs |
| Memory issues | Reduce max_length or batch size |
