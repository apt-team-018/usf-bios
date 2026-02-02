# SFT Alpaca Format

> **For USF-BIOS Frontend UI Training**

---

## Quick Start

Upload a `.jsonl` file with `instruction`, `input` (optional), and `output` fields.

```json
{"instruction": "Write a greeting", "input": "", "output": "Hello! How can I help you today?"}
```

---

## How to Use in USF-BIOS UI

### Step 1: Select Model
Choose your base model from the UI.

### Step 2: Configure Dataset
- Click **"Upload"** tab
- Upload your `.jsonl` file
- System auto-detects as **"SFT Alpaca"** format

### Step 3: Training Settings
| Setting | Value |
|---------|-------|
| **Training Method** | `SFT` (auto-selected) |
| **Train Type** | `LoRA` / `QLoRA` / `AdaLoRA` / `Full` |
| **Epochs** | 3 (recommended) |
| **Learning Rate** | 1e-5 to 2e-5 |

### Step 4: Review & Start
Click **"Start Training"** to begin.

---

## Format Structure

| Field | Required | Description |
|-------|----------|-------------|
| `instruction` | ✅ Yes | The task or question |
| `input` | ❌ No | Additional context (can be empty) |
| `output` | ✅ Yes | Expected model response |

**Auto-mapping:** USF-BIOS converts this to messages format internally:
- `instruction` + `input` → user message
- `output` → assistant response

---

## Examples

### Simple Instruction
```json
{"instruction": "What is 2+2?", "input": "", "output": "2+2 equals 4."}
```

### With Input Context
```json
{"instruction": "Summarize this text.", "input": "Machine learning enables computers to learn from data without explicit programming.", "output": "ML allows computers to learn from data automatically."}
```

### Code Generation
```json
{"instruction": "Write a Python function to reverse a string.", "input": "", "output": "def reverse_string(s):\n    return s[::-1]"}
```

### Translation
```json
{"instruction": "Translate to French.", "input": "Hello, how are you?", "output": "Bonjour, comment allez-vous?"}
```

### Classification
```json
{"instruction": "Classify the sentiment.", "input": "I love this product!", "output": "Positive"}
```

---

## When to Use Alpaca vs Messages

| Use Case | Recommended Format |
|----------|-------------------|
| Single-turn tasks | **Alpaca** ✅ |
| Multi-turn conversations | Messages |
| Tool calling | Messages |
| Existing Alpaca datasets | **Alpaca** ✅ |
| USF-Omega features | Messages |

**Note:** For advanced features (tool calling, self-reflection, model identity), use the **SFT Messages** format instead.

---

## Best Practices

1. **Clear Instructions** - Be specific and unambiguous
2. **Consistent Outputs** - Standardize response formatting
3. **Varied Tasks** - Mix different instruction types
4. **Quality Input** - Provide relevant context when needed

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Format not detected | Ensure `instruction` and `output` fields exist |
| Empty output error | Remove samples with empty `output` |
| Poor generalization | Add more diverse instructions |
