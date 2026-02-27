# GRPO + RLVR on MATH-500

## Overview

Replicate rasbt's GRPO + RLVR (Reinforcement Learning with Verifiable Rewards) approach for math reasoning, training on MATH dataset and evaluating on MATH-500 benchmark.

| Field | Value |
|-------|-------|
| **Reference** | rasbt's LLM reasoning fine-tuning (Chapter 6) |
| **Environment** | Mac (Apple Silicon MPS) / CUDA |
| **Base Model** | Qwen/Qwen3-0.6B |
| **Training Data** | math_full_minus_math500 (12k problems) |
| **Evaluation Data** | MATH-500 (500 held-out problems) |
| **Method** | GRPO with binary correctness rewards |

## Quick Start

```bash
# Setup
cd /Users/pramodsharma/code/RLCode
source venv/bin/activate
pip install -r requirements.txt

# Baseline evaluation
python scripts/eval_math500.py --model Qwen/Qwen3-0.6B

# Train with GRPO + RLVR
python scripts/train_grpo_rlvr.py --config configs/grpo_rlvr.yaml

# Evaluate trained model
python scripts/eval_math500.py --model checkpoints/grpo_rlvr/final
```

---

## Datasets

### Training Data

| Field | Value |
|-------|-------|
| **Source** | `rasbt/math_full_minus_math500` (GitHub) |
| **URL** | https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json |
| **Size** | 12,000 math problems |
| **Fields** | `problem`, `answer`, `solution`, `type`, `level`, `unique_id` |
| **Cache** | `data/math_train.json` |

### Evaluation Data

| Field | Value |
|-------|-------|
| **Source** | `HuggingFaceH4/MATH-500` |
| **Load** | `datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")` |
| **Size** | 500 problems (held-out test set) |
| **Fields** | `problem`, `answer`, `solution`, `subject`, `level`, `unique_id` |

### Subject Distribution (MATH-500)

| Subject | Count |
|---------|-------|
| Algebra | 124 |
| Intermediate Algebra | 97 |
| Prealgebra | 82 |
| Number Theory | 62 |
| Precalculus | 56 |
| Geometry | 41 |
| Counting & Probability | 38 |

### Difficulty Distribution (MATH-500)

| Level | Count |
|-------|-------|
| Level 1 | 43 |
| Level 2 | 90 |
| Level 3 | 105 |
| Level 4 | 128 |
| Level 5 | 134 |

---

## Model

| Field | Value |
|-------|-------|
| **Model** | `Qwen/Qwen3-0.6B` |
| **Parameters** | 0.6B (0.44B non-embedding) |
| **Architecture** | 28 layers, 16 attention heads |
| **Context Length** | 32K tokens |
| **License** | Apache 2.0 |
| **Fine-tuning** | Full (no LoRA) |
| **Precision** | float32 |

---

## RLVR Components

### Answer Extraction

Extracts answer from `\boxed{...}` format in model responses.

```python
from src.reasoning.math_grader import extract_boxed_answer

extract_boxed_answer("The answer is \\boxed{42}")  # Returns "42"
extract_boxed_answer("Therefore \\boxed{\\frac{1}{2}}")  # Returns "\\frac{1}{2}"
extract_boxed_answer("No boxed answer")  # Returns None
```

### Answer Grading

Compares predicted answer to ground truth with normalization.

| Comparison Type | Example |
|-----------------|---------|
| Exact match | `"42"` == `"42"` |
| Numeric equivalence | `"0.5"` == `"1/2"` |
| LaTeX fraction | `"\\frac{1}{2}"` == `"0.5"` |
| Case-insensitive | `"yes"` == `"YES"` |

### RLVR Reward Function

Binary reward based on correctness:

| Condition | Reward |
|-----------|--------|
| Correct answer in `\boxed{}` | 1.0 |
| Wrong answer in `\boxed{}` | 0.0 |
| No `\boxed{}` (format violation) | 0.0 |

```python
from src.reasoning.math_grader import reward_rlvr

reward_rlvr("The answer is \\boxed{42}", "42")  # Returns 1.0
reward_rlvr("The answer is \\boxed{42}", "43")  # Returns 0.0
reward_rlvr("No boxed answer", "42")  # Returns 0.0
```

---

## GRPO Algorithm

### Key Differences from Standard GRPO

| Aspect | Standard GRPO | RLVR GRPO |
|--------|---------------|-----------|
| **Reward** | Learned reward model | Binary correctness |
| **Log-prob** | Per-token, normalized | Sequence-level, SUMMED |
| **Loss** | Clipped surrogate (PPO-style) | Simple PG: `-(A * logp).mean()` |
| **Ratios** | `exp(new_lp - old_lp)` | No ratios, direct logprobs |
| **Clipping** | Yes (0.2 range) | No |
| **KL penalty** | Optional (adaptive) | Optional (fixed coefficient) |

### Algorithm (per step)

```
1. Sample a problem from training data
2. Generate N rollouts (responses) with temperature sampling
3. For each rollout:
   - Extract \boxed{} answer
   - Compare to ground truth
   - Assign binary reward (0 or 1)
4. Compute advantages: (rewards - mean) / (std + epsilon)
5. Compute SUMMED log-probs for each rollout
6. Loss = -(advantages.detach() * logps).mean() + kl_penalty (optional)
7. Backprop and update
```

### Why Summed Log-Probs?

Rasbt uses **summed** (not averaged) sequence-level log probabilities:

1. Averaging would rescale the learning signal by sequence length
2. Summed logprobs encourage shorter responses (more negative terms for longer sequences)
3. Provides stronger gradient signal

```python
logp = sum(log p(token_t | tokens<t, prompt)) for t in response tokens
```

### Advantage Normalization

Binary rewards become continuous advantages:

```
Example with 8 rollouts:
Rewards = [0, 0, 1, 0, 0, 1, 0, 0]  # 2 correct, 6 wrong
mean = 0.25, std = 0.433

Advantages:
- Wrong (reward=0): (0 - 0.25) / 0.433 = -0.577
- Correct (reward=1): (1 - 0.25) / 0.433 = +1.732
```

### Edge Cases

| Scenario | Advantages | Effect |
|----------|------------|--------|
| All wrong `[0,0,0,0]` | `[0,0,0,0]` | No gradient (skipped) |
| All correct `[1,1,1,1]` | `[0,0,0,0]` | No gradient (skipped) |
| One correct `[0,0,0,1]` | Large positive for correct | Strong learning signal |

---

## Hyperparameters

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Rollouts per prompt** | 8 | Recommended minimum |
| **Max new tokens** | 512 | Required for full reasoning |
| **Temperature** | 0.8 | For diverse sampling |
| **Top-p** | 0.9 | Nucleus sampling |
| **Learning rate** | 1e-5 | |
| **Gradient clipping** | 1.0 | Max gradient norm |
| **Optimizer** | AdamW | |
| **Training steps** | 50 | |
| **Checkpoint frequency** | Every 5 steps | |

### KL Penalty (Optional)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Enabled** | No (default) | Chapter 6 baseline |
| **Coefficient** | 0.02 | When enabled |
| **Normalization** | Per-token | Prevents length scaling |

---

## File Structure

```
RLCode/
├── src/
│   ├── data/
│   │   └── math_loader.py          # Load MATH-500 eval + MATH train
│   └── reasoning/
│       └── math_grader.py          # extract_boxed, normalize, grade_answer
├── scripts/
│   ├── eval_math500.py             # Evaluate any model on MATH-500
│   └── train_grpo_rlvr.py          # GRPO with verifiable rewards
├── configs/
│   └── grpo_rlvr.yaml              # Training configuration
├── checkpoints/
│   └── grpo_rlvr/                  # Saved model checkpoints
├── results/
│   └── math500_eval_*.json         # Evaluation results
└── data/
    └── math_train.json             # Cached training data
```

---

## Expected Results

### Baseline (Qwen3-0.6B)

| Metric | Expected |
|--------|----------|
| Overall Accuracy | ~15% |
| Format Compliance | ~60-80% |

### After Training (50 steps)

| Metric | Expected |
|--------|----------|
| Overall Accuracy | ~47% |
| Format Compliance | ~90%+ |
| Improvement | +30% absolute |

---

## Evaluation Metrics

### Primary Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % problems solved correctly |
| **Format Compliance** | % responses with `\boxed{}` |

### Breakdown Metrics

| Breakdown | Description |
|-----------|-------------|
| **By Subject** | Accuracy per math subject (Algebra, Geometry, etc.) |
| **By Level** | Accuracy per difficulty level (1-5) |
| **Response Length** | Average tokens generated |

### Evaluation Output

```
============================================================
MATH-500 Evaluation Results
============================================================

Model: Qwen/Qwen3-0.6B
Total: 500 problems

Overall Accuracy: 78/500 = 15.6%

By Subject:
  Algebra              23/124 = 18.5%
  Geometry              8/ 41 = 19.5%
  Number Theory        12/ 62 = 19.4%
  ...

By Difficulty Level:
  Level 1              15/ 43 = 34.9%
  Level 2              20/ 90 = 22.2%
  Level 3              18/105 = 17.1%
  Level 4              15/128 = 11.7%
  Level 5              10/134 =  7.5%

Format Compliance: 400/500 = 80.0%
Avg Response Length: 245 tokens
============================================================
```

---

## Memory Requirements

### Qwen3-0.6B Full Fine-tuning (float32)

| Component | Memory |
|-----------|--------|
| Model parameters | ~1.2 GB |
| Optimizer states (AdamW) | ~2.4 GB |
| Gradients | ~1.2 GB |
| Activations (512 tokens) | ~0.5-1 GB |
| **Total estimate** | **~5-6 GB** |

### If OOM Occurs

1. Reduce `num_rollouts` (8 → 4 → 2)
2. Reduce `max_new_tokens` (512 → 256)
3. Try float16/bfloat16
4. Enable gradient checkpointing

---

## Commands Reference

### Evaluation

```bash
# Baseline evaluation
python scripts/eval_math500.py --model Qwen/Qwen3-0.6B

# Evaluate subset (faster)
python scripts/eval_math500.py --model Qwen/Qwen3-0.6B --max-problems 100

# Batched evaluation (faster)
python scripts/eval_math500.py --model Qwen/Qwen3-0.6B --batch-size 8

# Verbose output
python scripts/eval_math500.py --model Qwen/Qwen3-0.6B --verbose

# Evaluate trained model
python scripts/eval_math500.py --model checkpoints/grpo_rlvr/step_50
```

### Training

```bash
# Train with config file
python scripts/train_grpo_rlvr.py --config configs/grpo_rlvr.yaml

# Train with CLI args
python scripts/train_grpo_rlvr.py --num-steps 50 --num-rollouts 8

# Train with KL penalty
python scripts/train_grpo_rlvr.py --use-kl-penalty --kl-coeff 0.02

# Quick test run
python scripts/train_grpo_rlvr.py --num-steps 5 --num-rollouts 4 --max-new-tokens 256
```

---

## Verification Checklist

- [ ] **Data loading**: Verify MATH-500 loads 500 problems
- [ ] **Data loading**: Verify training data loads ~12k problems
- [ ] **Answer extraction**: Test `\boxed{}` parsing on edge cases
- [ ] **Grading**: Test numeric equivalence (0.5 == 1/2)
- [ ] **Baseline eval**: Run on base model (expect ~15%)
- [ ] **Training**: Run 50 steps, check reward distribution
- [ ] **Final eval**: Run on trained model (target ~47%)

---

## References

- rasbt's LLM fine-tuning repository (Chapter 6: GRPO + RLVR)
- HuggingFace MATH-500 dataset
- Original MATH dataset (Hendrycks et al.)
