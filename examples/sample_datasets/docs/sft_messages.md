# SFT Messages Format

> **For USF-BIOS Frontend UI Training**

---

## Quick Start

Upload a `.jsonl` file where each line contains a `messages` array with conversations.

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help you?"}]}
```

---

## How to Use in USF-BIOS UI

### Step 1: Select Model
In the UI, choose your base model. The system auto-detects if it's a text, vision, audio, or video model.

### Step 2: Configure Dataset
- Click **"Upload"** tab
- Upload your `.jsonl` file
- System auto-detects format as **"SFT"**

### Step 3: Training Settings
| Setting | Value |
|---------|-------|
| **Training Method** | `SFT` (auto-selected) |
| **Train Type** | `LoRA` / `QLoRA` / `AdaLoRA` / `Full` |
| **Epochs** | 3 (recommended start) |
| **Learning Rate** | 1e-4 to 2e-4 |
| **Batch Size** | 1-4 (GPU dependent) |

### Step 4: Review & Start
Click **"Start Training"** to begin.

---

## Supported Roles

| Role | Description | Trained? |
|------|-------------|----------|
| `system` | System instructions (becomes developer block) | ❌ No |
| `developer` | Developer instructions | ❌ No |
| `user` | User messages | ❌ No |
| `assistant` | Model responses | ✅ **Yes** |
| `tool` | Tool/function responses | ❌ No |

**Only `assistant` content is trained.** All other roles provide context.

---

## Basic Examples

### Simple Conversation
```json
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI (Artificial Intelligence) is the simulation of human intelligence by machines."}]}
```

### With System Prompt
```json
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Write hello world in Python"}, {"role": "assistant", "content": "print('Hello, World!')"}]}
```

### Multi-turn Conversation
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well, thank you!"}]}
```

---

## Advanced: USF-Omega Features

If training USF-Omega models, you can use these advanced features:

### Model Identity
Use `model_identity` for the actual system identity (separate from developer instructions):

```json
{
  "model_identity": "I am USF Omega, an AI assistant by Ultrasafe AI.",
  "messages": [
    {"role": "system", "content": "Always be helpful and accurate."},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am USF Omega, an AI assistant created by Ultrasafe AI."}
  ]
}
```

### Developer Role
Use `developer` role for instructions separate from system:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "developer", "content": "Use formal language. Respond in JSON when asked."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Good day. How may I assist you?"}
  ]
}
```

### Function/Tool Calling
Train models to use tools with `tools` and `tool_calls`:

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in NYC?"},
    {
      "role": "assistant",
      "content": "I'll check the weather for you.",
      "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "NYC"}}}]
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "Sunny, 72°F"},
    {"role": "assistant", "content": "The weather in NYC is sunny at 72°F!"}
  ],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
}
```

### Parallel Tool Calls
Multiple tools called at once:

```json
{
  "messages": [
    {"role": "user", "content": "Search for A and B"},
    {
      "role": "assistant",
      "content": "Searching both.",
      "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "search", "arguments": {"q": "A"}}},
        {"id": "c2", "type": "function", "function": {"name": "search", "arguments": {"q": "B"}}}
      ]
    }
  ],
  "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
}
```

### Structured JSON Output
Use `response_schema` for structured responses:

```json
{
  "messages": [
    {"role": "user", "content": "Get user info"},
    {"role": "assistant", "content": "{\"name\": \"John\", \"age\": 30}"}
  ],
  "response_schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}
}
```

### JSON Mode
Simple JSON output:

```json
{
  "messages": [
    {"role": "user", "content": "List 3 colors"},
    {"role": "assistant", "content": "{\"colors\": [\"red\", \"blue\", \"green\"]}"}
  ],
  "json_mode": true
}
```

### Self-Reflection
Train models to reflect on their responses:

```json
{
  "messages": [
    {"role": "user", "content": "Write a haiku"},
    {
      "role": "assistant",
      "content": "Cherry blossoms fall\nGentle breeze carries petals\nSpring whispers goodbye",
      "self_reflection": "Good imagery and 5-7-5 syllable structure maintained."
    }
  ],
  "self_reflection_enabled": true
}
```

### Contents Array (Multi-iteration)
For complex responses with multiple steps:

```json
{
  "messages": [
    {"role": "user", "content": "Research and summarize"},
    {
      "role": "assistant",
      "contents": [
        {"type": "text", "text": "Let me search for that..."},
        {"type": "tool_calls", "tool_calls": [{"id": "s1", "type": "function", "function": {"name": "search", "arguments": {"q": "topic"}}}]},
        {"type": "self_reflection", "self_reflection": "Found good results."},
        {"type": "text", "text": "Based on my research, here's the summary..."}
      ]
    }
  ],
  "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
  "self_reflection_enabled": true
}
```

---

## Multimodal Training

### Vision (Images)
For vision models, add `images` array:

```json
{
  "messages": [
    {"role": "user", "content": "What's in this image?"},
    {"role": "assistant", "content": "The image shows a golden retriever playing in a park."}
  ],
  "images": ["path/to/dog.jpg"]
}
```

### Audio
For audio models, add `audios` array:

```json
{
  "messages": [
    {"role": "user", "content": "Transcribe this audio."},
    {"role": "assistant", "content": "The speaker says: Welcome to the presentation."}
  ],
  "audios": ["path/to/audio.mp3"]
}
```

### Video
For video models, add `videos` array:

```json
{
  "messages": [
    {"role": "user", "content": "Describe this video."},
    {"role": "assistant", "content": "The video shows a cooking tutorial for pasta."}
  ],
  "videos": ["path/to/video.mp4"]
}
```

---

## Feature Flags Summary

| Field | Type | Description |
|-------|------|-------------|
| `model_identity` | string | Model's identity (actual system block) |
| `tools` | array | Function/tool definitions |
| `response_schema` | object | JSON Schema for structured output |
| `json_mode` | boolean | Force JSON output |
| `self_reflection_enabled` | boolean | Enable self-reflection training |
| `search_enabled` | boolean | Enable built-in search function |
| `citations_enabled` | boolean | Enable citation format |

---

## Best Practices

1. **Diverse Topics** - Include varied subjects
2. **Consistent Style** - Maintain uniform response formatting
3. **Quality Over Quantity** - High-quality samples matter more
4. **Balance Length** - Mix short and detailed responses
5. **Real Conversations** - Mimic actual user interactions

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Format not detected | Ensure each line has `messages` array |
| Empty responses | Remove samples with empty `assistant` content |
| Invalid role | Use only: `system`, `developer`, `user`, `assistant`, `tool` |
| Tool calls not working | Ensure `tools` array is defined with `tool_calls` |
