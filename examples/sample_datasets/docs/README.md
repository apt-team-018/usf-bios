# USF-BIOS Dataset Format Documentation

**Version:** 1.0.0  
**Last Updated:** February 2026  
**Compatible with:** USF-BIOS Training Framework

---

## Quick Reference

| Format | Training Method | Algorithms | Best For |
|--------|-----------------|------------|----------|
| [SFT Messages](./sft_messages.md) | SFT | - | Chatbots, instruction following |
| [SFT Alpaca](./sft_alpaca.md) | SFT | - | Task-specific training |
| [Pre-Training](./pt_raw_text.md) | PT | - | Domain adaptation |
| [RLHF Simple](./rlhf_pref_simple.md) | RLHF | DPO, ORPO, SimPO, CPO, RM | Preference alignment |
| [RLHF Messages](./rlhf_pref_messages.md) | RLHF | DPO, ORPO, SimPO, CPO, RM | Multi-turn preferences |
| [RLHF KTO](./rlhf_kto.md) | RLHF | KTO | Binary feedback |
| [RLHF Online](./rlhf_online.md) | RLHF | PPO, GRPO, GKD | Online RL training |

---

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    USF-BIOS Training Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Stage 1: Pre-Training (Optional)                              │
│   ┌─────────────┐                                               │
│   │  Raw Text   │ ──▶ Domain Knowledge Injection                │
│   │  (pt_*)     │                                               │
│   └─────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│   Stage 2: Supervised Fine-Tuning                               │
│   ┌─────────────┐                                               │
│   │  Messages   │ ──▶ Instruction Following                     │
│   │  or Alpaca  │                                               │
│   └─────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│   Stage 3: RLHF Alignment (Optional)                            │
│   ┌─────────────┐                                               │
│   │ Preference  │ ──▶ Human Preference Alignment                │
│   │ or KTO/PPO  │                                               │
│   └─────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Format Selection Guide

### By Use Case

| Use Case | Recommended Format | Training Method |
|----------|-------------------|-----------------|
| Build a chatbot | SFT Messages | SFT |
| Task-specific assistant | SFT Alpaca | SFT |
| Add domain knowledge | Pre-Training Raw Text | PT |
| Improve response quality | RLHF Simple/Messages | RLHF (DPO) |
| Use thumbs up/down data | RLHF KTO | RLHF (KTO) |
| Maximize custom reward | RLHF Online | RLHF (PPO/GRPO) |

### By Data Availability

| You Have | Use Format | Algorithm |
|----------|-----------|-----------|
| Conversations with good responses | SFT Messages | SFT |
| Instruction-output pairs | SFT Alpaca | SFT |
| Raw text documents | Pre-Training | PT |
| Chosen vs rejected pairs | RLHF Simple | DPO/ORPO |
| Good/bad labels | RLHF KTO | KTO |
| Prompts + reward model | RLHF Online | PPO/GRPO |

---

## Training Type Compatibility

All formats support these training types:

| Training Type | Description | Memory | Quality |
|---------------|-------------|--------|---------|
| **LoRA** | Low-Rank Adaptation | Low | Good |
| **QLoRA** | Quantized LoRA | Very Low | Good |
| **AdaLoRA** | Adaptive LoRA | Low | Better |
| **Full** | Full parameter tuning | High | Best |

### Recommendations

- **Limited GPU (< 24GB)**: Use QLoRA
- **Standard GPU (24-48GB)**: Use LoRA
- **Multi-GPU / Cloud**: Use Full fine-tuning for best results

---

## RLHF Algorithm Reference

### Offline Algorithms (Pre-collected Data)

| Algorithm | Data Format | Stability | Compute | Best For |
|-----------|-------------|-----------|---------|----------|
| **DPO** | Preference pairs | High | Low | General alignment |
| **ORPO** | Preference pairs | High | Low | No reference model |
| **SimPO** | Preference pairs | High | Low | Simplicity |
| **CPO** | Preference pairs | Medium | Low | Strong contrast |
| **KTO** | Binary labels | High | Low | Thumbs up/down data |
| **RM** | Preference pairs | High | Low | Reward model training |

### Online Algorithms (Generated Data)

| Algorithm | Requirements | Stability | Compute | Best For |
|-----------|--------------|-----------|---------|----------|
| **PPO** | Reward model | Medium | High | General RL |
| **GRPO** | Reward model | Higher | Medium | Group comparisons |
| **GKD** | Teacher model | High | Medium | Distillation |

---

## File Format Requirements

### JSONL Format (Recommended)

All datasets should use JSONL (JSON Lines) format:
- One JSON object per line
- UTF-8 encoding
- No trailing commas

```jsonl
{"field1": "value1", "field2": "value2"}
{"field1": "value3", "field2": "value4"}
```

### Supported File Types

| Extension | Max Size | Best For |
|-----------|----------|----------|
| `.jsonl` | Unlimited | Large datasets |
| `.json` | 2GB | Small datasets |
| `.csv` | Unlimited | Tabular data |
| `.txt` | Unlimited | Raw text (PT only) |

---

## Data Quality Checklist

### Before Training

- [ ] **Format validated**: All required fields present
- [ ] **No empty fields**: Remove samples with empty content
- [ ] **Consistent encoding**: UTF-8 throughout
- [ ] **Deduplicated**: Remove exact duplicates
- [ ] **Balanced**: Check label/category distribution
- [ ] **Diverse**: Cover target use cases

### For RLHF Data

- [ ] **Clear preference**: Chosen is objectively better
- [ ] **Same context**: User messages match in pairs
- [ ] **Quality difference**: Not too obvious, not too subtle
- [ ] **Balanced pairs**: ~50/50 for KTO labels

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "Invalid JSON" | Malformed data | Validate each line with `json.loads()` |
| "Missing field" | Incomplete samples | Filter samples without required fields |
| "Loss not decreasing" | Data quality | Check for duplicates, improve diversity |
| "OOM error" | Batch too large | Reduce batch size or use QLoRA |
| "Poor results" | Wrong format | Verify format matches training method |

---

## Sample Datasets

Download example datasets for each format:

| Format | File | Samples | Size |
|--------|------|---------|------|
| SFT Messages | `sft_messages.jsonl` | 5 | 5.6 KB |
| SFT Alpaca | `sft_alpaca.jsonl` | 5 | 3.6 KB |
| Pre-Training | `pt_raw_text.jsonl` | 10 | 3.7 KB |
| RLHF Simple | `rlhf_offline_preference.jsonl` | 5 | 6.1 KB |
| RLHF Messages | `rlhf_pref_messages.jsonl` | 5 | 6.2 KB |
| RLHF KTO | `kto_messages.jsonl` | 8 | 4.0 KB |
| RLHF Online | `rlhf_online_prompt.jsonl` | 10 | 0.7 KB |

---

## Getting Help

- **Documentation**: Each format has detailed docs (linked above)
- **Examples**: Sample datasets in `/examples/sample_datasets/`
- **Validation**: Use the dataset type detection API to validate your data

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Feb 2026 | Initial documentation release |
