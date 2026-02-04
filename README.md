# Reward Model: Qwen 2.5 3B Fine-Tuned on Anthropic RLHF

This repository contains a **Reward Model** based on **Qwen 2.5 3B**, fine-tuned on the **Anthropic RLHF** dataset using `trl.RewardTrainer`. The model is designed to score completions given a prompt, which is useful for reinforcement learning from human feedback (RLHF) pipelines and evaluation tasks.

The model is available on **Hugging Face**: [https://huggingface.co/kanishkez/Reward-Model](https://huggingface.co/kanishkez/Reward-Model)

---

## Model Overview

- **Base Model:** Qwen 2.5 3B
- **Fine-Tuned On:** Anthropic RLHF dataset
- **Output:** Single scalar reward score per prompt-completion pair
- **Framework:** PyTorch + Transformers + TRL
- **Model Type:** Reward Model for RLHF
- **Language:** English (primarily)

---

## Installation

Install the required dependencies:

```bash
pip install torch transformers datasets trl
```

For GPU acceleration (recommended):

```bash
pip install torch transformers datasets trl accelerate
```

---

## Usage

### Quick Start

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "kanishkez/Reward-Model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare input
prompt = "What is the capital of France?"
completion = "The capital of France is Paris."
input_text = f"{prompt}\n{completion}"

# Tokenize and get reward score
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    reward_score = outputs.logits[0].item()

print(f"Reward Score: {reward_score:.4f}")
```

### Training

To train the model from scratch:

```bash
python trainer.py
```

**Training Configuration:**
- Learning rate: Configured in `trainer.py`
- Batch size: Optimized for available GPU memory
- Epochs: Specified in training script
- Dataset: Anthropic RLHF dataset

### Inference

To run inference on the trained model:

```bash
python inference.py
```

---

## RewardBench Evaluation

The model was evaluated using [RewardBench](https://github.com/allenai/reward-bench), a comprehensive benchmark for reward models. Results:

| Category | Score |
|----------|-------|
| **Chat** | 83.5% |
| **Chat Hard** | 53.2% |
| **Safety** | 72.2% |
| **Reasoning** | 73.4% |

<img width="1200" height="735" alt="image" src="https://github.com/user-attachments/assets/bd6cb251-b939-464a-a023-b047e08c910a" />



---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

