# RLHF Preference - Messages Format

> **For USF-BIOS Frontend UI Training**

---

## Quick Start

Upload a `.jsonl` file with `messages` (chosen) and `rejected_messages` arrays.

```json
{
  "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help?"}],
  "rejected_messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "What?"}]
}
```

---

## How to Use in USF-BIOS UI

### Step 1: Select Model
Choose your base model (should be SFT-trained first).

### Step 2: Configure Dataset
- Click **"Upload"** tab
- Upload your `.jsonl` file
- System auto-detects as **"RLHF Messages Preference"** format

### Step 3: Training Settings
| Setting | Value |
|---------|-------|
| **Training Method** | `RLHF` (auto-selected) |
| **RLHF Algorithm** | `DPO` / `ORPO` / `SimPO` / `CPO` / `RM` |
| **Train Type** | `LoRA` / `QLoRA` / `AdaLoRA` / `Full` |
| **Beta** | 0.1 (default) |
| **Batch Size** | 2 (lower due to longer sequences) |

### Step 4: Review & Start
Click **"Start Training"** to begin.

---

## Format Structure

| Field | Required | Description |
|-------|----------|-------------|
| `messages` | ✅ Yes | Conversation with **preferred** assistant response |
| `rejected_messages` | ✅ Yes | Conversation with **non-preferred** response |

**Important:** User messages must be identical in both arrays. Only the assistant response differs.

---

## Examples

### Single-turn
```json
{
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ],
  "rejected_messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "I don't know."}
  ]
}
```

### With System Prompt
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain AI"},
    {"role": "assistant", "content": "AI is artificial intelligence - machines that simulate human cognitive functions."}
  ],
  "rejected_messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain AI"},
    {"role": "assistant", "content": "AI is robots."}
  ]
}
```

### Multi-turn
```json
{
  "messages": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm doing well, thanks for asking! How can I help you today?"}
  ],
  "rejected_messages": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "Fine."}
  ]
}
```

### Vision (Multimodal)
```json
{
  "messages": [
    {"role": "user", "content": "Describe this image"},
    {"role": "assistant", "content": "The image shows a sunset over the ocean with orange and pink clouds."}
  ],
  "rejected_messages": [
    {"role": "user", "content": "Describe this image"},
    {"role": "assistant", "content": "It's a picture."}
  ],
  "images": ["path/to/sunset.jpg"]
}
```

---

## When to Use Messages vs Simple Format

| Use Case | Recommended |
|----------|-------------|
| Multi-turn conversations | **Messages** ✅ |
| System prompts needed | **Messages** ✅ |
| Simple Q&A pairs | Simple |
| Existing HuggingFace datasets | Simple |

---

## Best Practices

1. **Same User Content** - User messages must be identical in both arrays
2. **Same System Prompts** - Keep system prompts identical
3. **Only Assistant Differs** - Only the assistant's final response should differ
4. **Clear Preference** - Chosen should be objectively better

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Format not detected | Ensure both `messages` and `rejected_messages` exist |
| Mismatched count | Both arrays must have same number of messages |
| User messages differ | Only assistant content should differ |
| OOM errors | Reduce batch_size or max_length |

