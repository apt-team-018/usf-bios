# Iterative Self-Training Dataset Format

## Overview

Iterative self-training (also known as ReST, STaR, or Expert Iteration) is a powerful technique where:
1. **Generate**: Model generates responses to prompts
2. **Judge**: Responses are scored by a reward model or verification function
3. **Filter**: Best responses are selected
4. **Train**: Model is fine-tuned on the filtered data
5. **Repeat**: Process continues for multiple rounds

This approach enables continuous self-improvement without human annotation.

## Dataset Formats

### Basic Prompts Format

The simplest format - just prompts that the model will respond to:

```json
{"prompt": "Explain photosynthesis in simple terms."}
{"prompt": "Write a Python function to sort a list."}
{"prompt": "Solve: What is 15% of 240?"}
```

### Prompts with Difficulty (Curriculum Learning)

Add difficulty levels to enable curriculum-based training:

```json
{"prompt": "What is 2 + 2?", "difficulty": "easy"}
{"prompt": "Solve: 3x + 7 = 22", "difficulty": "medium"}
{"prompt": "Prove that √2 is irrational", "difficulty": "hard"}
{"prompt": "Solve the integral: ∫(x²·sin(x))dx", "difficulty": "expert"}
```

**Difficulty Levels:**
- `easy` - Simple tasks, high success rate expected
- `medium` - Moderate complexity
- `hard` - Challenging tasks
- `expert` - Complex reasoning required

### Prompts with Verifiable Answers (Rule-Based Scoring)

For math, coding, or other verifiable tasks, include expected answers:

```json
{
  "prompt": "Calculate: 15 × 8 + 32",
  "difficulty": "easy",
  "metadata": {
    "expected_answer": "152",
    "category": "arithmetic"
  }
}
```

### Code Tasks with Test Cases

```json
{
  "prompt": "Write a Python function that returns the nth Fibonacci number.",
  "difficulty": "medium",
  "metadata": {
    "test_cases": [
      {"input": "print(fib(0))", "expected_output": "0"},
      {"input": "print(fib(1))", "expected_output": "1"},
      {"input": "print(fib(10))", "expected_output": "55"}
    ],
    "category": "coding"
  }
}
```

### Multi-Turn Conversation Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "Help me understand quadratic equations."}
  ],
  "difficulty": "medium",
  "metadata": {
    "category": "education"
  }
}
```

## Reward Model Options

### 1. Local Reward Model (Default)

Load a HuggingFace reward model locally:

```json
{
  "reward_config": {
    "type": "local",
    "model_path": "OpenAssistant/reward-model-deberta-v3-large-v2"
  }
}
```

### 2. External API

Send responses to an external scoring service:

```json
{
  "reward_config": {
    "type": "api",
    "api_endpoint": "https://your-server.com/score",
    "api_key": "your-api-key",
    "api_timeout": 30.0,
    "api_batch_size": 10
  }
}
```

**API Request Format:**
```json
{
  "items": [
    {"prompt": "...", "response": "..."},
    {"prompt": "...", "response": "..."}
  ]
}
```

**API Response Format:**
```json
{
  "scores": [0.85, 0.72]
}
```

### 3. Custom Script

Run a custom Python script for scoring:

```json
{
  "reward_config": {
    "type": "script",
    "script_path": "/path/to/my_scorer.py",
    "script_function": "score"
  }
}
```

**Script Template:**
```python
# my_scorer.py
import json
import argparse

def score(input_file: str, output_file: str):
    """Score generated responses."""
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Your scoring logic here
            score = calculate_score(item['prompt'], item['response'])
            item['reward_score'] = score
            results.append(item)
    
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--function', default='score')
    args = parser.parse_args()
    
    if args.function == 'score':
        score(args.input, args.output)
```

### 4. Rule-Based Verification

For verifiable tasks (math, code, JSON, regex):

```json
{
  "reward_config": {
    "type": "rule_based",
    "rule_type": "math"
  }
}
```

**Rule Types:**
- `math` - Extracts numerical answer and compares to expected
- `code` - Executes code and checks test cases
- `json` - Validates JSON format
- `regex` - Matches response against regex pattern

## Dataset Selection Strategies

### Sequential (Default)

Process all prompts in order each round:

```json
{
  "dataset_selection_strategy": "sequential",
  "samples_per_round": 1000
}
```

### Difficulty Curriculum

Start easy, increase difficulty as rounds progress:

```json
{
  "dataset_selection_strategy": "difficulty_curriculum",
  "enable_difficulty_curriculum": true,
  "difficulty_progression": ["easy", "easy", "medium", "medium", "hard", "hard", "expert"]
}
```

### Round Robin

Alternate between multiple datasets:

```json
{
  "datasets": [
    {"path": "/data/math.jsonl", "name": "Math", "difficulty": "medium"},
    {"path": "/data/code.jsonl", "name": "Coding", "difficulty": "medium"},
    {"path": "/data/writing.jsonl", "name": "Writing", "difficulty": "easy"}
  ],
  "dataset_selection_strategy": "round_robin"
}
```

### Random Weighted

Random selection with configurable weights:

```json
{
  "datasets": [
    {"path": "/data/easy.jsonl", "weight": 3.0},
    {"path": "/data/medium.jsonl", "weight": 2.0},
    {"path": "/data/hard.jsonl", "weight": 1.0}
  ],
  "dataset_selection_strategy": "random_weighted"
}
```

## Complete Configuration Example

```json
{
  "name": "math-improvement-training",
  "base_model_path": "/models/llama-8b-math",
  
  "reward_config": {
    "type": "rule_based",
    "rule_type": "math"
  },
  
  "datasets": [
    {
      "path": "/data/easy_math.jsonl",
      "name": "Easy Math",
      "difficulty": "easy",
      "weight": 1.0
    },
    {
      "path": "/data/hard_math.jsonl",
      "name": "Hard Math",
      "difficulty": "hard",
      "weight": 1.0
    }
  ],
  
  "dataset_selection_strategy": "difficulty_curriculum",
  "enable_difficulty_curriculum": true,
  "samples_per_round": 500,
  
  "num_rounds": 10,
  "num_generations_per_prompt": 8,
  "filter_strategy": "top_k_percent",
  "filter_top_k_percent": 20.0,
  
  "training_method": "lora",
  "learning_rate": 1e-5,
  "num_train_epochs": 1
}
```

## Best Practices

### 1. Start Small
- Begin with 5-10 rounds to validate your setup
- Use a small sample (100-500 prompts) initially

### 2. Use Appropriate Reward Strategy
- **Verifiable tasks** (math, code): Use `rule_based`
- **Open-ended tasks**: Use `local` reward model or `api`
- **Custom domains**: Create a `script`

### 3. Curriculum Learning
- Start with easy problems to build confidence
- Gradually increase difficulty
- Monitor mean reward scores per round

### 4. Filter Wisely
- `top_k_percent`: Keep top 10-30% (stricter = higher quality)
- `threshold`: Keep samples above a score (e.g., 0.7)
- `best_of_n`: Keep best response per prompt

### 5. Monitor Training
- Check round metrics (mean score, loss)
- Stop if scores plateau or decline
- Save checkpoints to recover from failures

## Troubleshooting

### Low Reward Scores
- Check if reward model/rules are appropriate
- Verify prompt format matches model expectations
- Try lower temperature for more deterministic outputs

### Memory Issues
- Reduce `num_generations_per_prompt`
- Use smaller batch sizes
- Enable VRAM cleanup between phases

### API Timeouts
- Increase `api_timeout`
- Reduce `api_batch_size`
- Implement retry logic in your API

## Related Methods

- **SFT**: Supervised fine-tuning on curated data
- **DPO**: Direct preference optimization with pairs
- **PPO**: Online RLHF with real-time generation
- **GRPO**: Group relative policy optimization

Iterative training is complementary to these methods and can be combined for even better results.
