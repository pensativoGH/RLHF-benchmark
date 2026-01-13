# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

Reinforcement learning experiments for LLM training and alignment. Focus areas:
- **Reward Modeling** - Training reward models from preference data
- **DPO** - Direct Preference Optimization (reference-free alignment)
- **PPO** - Proximal Policy Optimization for RLHF
- **GRPO** - Group Relative Policy Optimization
- **Reasoning RL** - RL techniques for improving reasoning capabilities

## Project Structure

```
RLCode/
├── reward_model/       # Reward model training
├── dpo/                # DPO experiments
├── ppo/                # PPO-based RLHF
├── grpo/               # GRPO experiments
├── reasoning/          # Reasoning-focused RL
├── data/               # Preference datasets, prompts
├── configs/            # Training configurations
├── scripts/            # Training and evaluation scripts
└── results/            # Checkpoints, logs, metrics
```

## Setup

```bash
cd /Users/pramodsharma/code/RLCode
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Key Libraries

- **TRL** (Transformers Reinforcement Learning) - Hugging Face library for RLHF, DPO, PPO
- **DeepSpeed** - Distributed training and memory optimization
- **PEFT** - Parameter-efficient fine-tuning (LoRA, QLoRA)
- **vLLM** - Fast inference for generation
- **Weights & Biases** - Experiment tracking

## Common Commands

```bash
# Activate environment
source venv/bin/activate

# Train reward model
python scripts/train_reward_model.py --config configs/reward_model.yaml

# DPO training
python scripts/train_dpo.py --config configs/dpo.yaml

# PPO training
python scripts/train_ppo.py --config configs/ppo.yaml

# Evaluate model
python scripts/evaluate.py --model checkpoints/model --benchmark <benchmark>
```

## Datasets

### Small Datasets (Mac-friendly, <10K samples)
- **Dahoas/synthetic-instruct-gptj-pairwise** - ~33K pairs, can subset
- **lvwerra/stack-exchange-paired** - Can filter by score, subset by topic
- **CarperAI/openai_summarize_comparisons** - ~93K, use small subset
- **argilla/ultrafeedback-binarized-preferences-cleaned** - Filter to small subset
- **tatsu-lab/alpaca_farm** - ~10K preference pairs
- **HuggingFaceH4/helpful-instructions** - Smaller helpful dataset
- **jondurbin/airoboros-gpt4-1.4.1** - Can subset for quick experiments

### Medium Datasets (Cloud/GPU recommended)
- Anthropic HH-RLHF (~170K)
- OpenAssistant conversations
- UltraFeedback (~64K)

### Tips for Mac Training
- Use small models: TinyLlama, Phi-2, Qwen-0.5B, SmolLM
- Enable MPS backend: `device = "mps"`
- Use 4-bit quantization with bitsandbytes
- LoRA/QLoRA for memory efficiency
- Batch size 1-2, gradient accumulation
- Subset datasets to 1K-5K samples for quick iterations

## Notes

- Track all hyperparameters and random seeds
- Log KL divergence from reference model
- Monitor reward hacking and over-optimization
- Use evaluation benchmarks: MT-Bench, AlpacaEval, etc.
