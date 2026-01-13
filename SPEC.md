# RLCode Experiment Spec

## Overview

Reinforcement Learning experiments for LLM training and alignment.

- **Environment:** Mac (Apple Silicon MPS)
- **Base Model:** Qwen2-0.5B
- **Primary Dataset:** Anthropic HH-RLHF (10k sample subset)

## Summary

### Core Milestones
| # | Milestone | Dataset | Key Output |
|---|-----------|---------|------------|
| M1 | Reward Model | Anthropic HH-RLHF 10k | `reward_model.score()` |
| M2 | DPO | Anthropic HH-RLHF 10k | Aligned model, win rate |
| M3 | PPO | HH-RLHF prompts + M1 reward | RLHF-aligned model |
| M4 | GRPO | HH-RLHF prompts + M1 reward | Reference-free aligned model |
| M5 | Process Reward Model | Math-Shepherd 10k | `prm.score_steps()` |
| M6 | MCTS + LLM | GSM8K 1k test | `mcts.solve()`, solve rates |

### Experiments
| # | Experiment | Description |
|---|------------|-------------|
| A | DPO vs PPO | Compare stability, compute cost |
| B | β Ablation | DPO beta: 0.01, 0.1, 1.0 |
| C | KL vs Reward | Track reward hacking |
| D | Value Head | Frozen vs unfrozen in PPO |
| E | Reward Shaping | Length, format, repetition penalties |
| F | GRPO vs PPO | Compute efficiency, stability |
| G | Group Size | GRPO: 4 vs 8 vs 16 |
| H | GRPO + DPO | Hybrid regularization |
| I | PRM vs ORM | Process vs outcome supervision |
| J | DPO vs PPO vs GRPO | Head-to-head on 100 held-out test examples |

---

## Milestone 1: Reward Model Training

**Objective:** Train a reward model on preference data to score response quality

**Dataset:**
| Field | Value |
|-------|-------|
| Source | `Anthropic/hh-rlhf` (HuggingFace) |
| Size | 10,000 samples (chosen/rejected pairs) |
| Split | 8k train / 1k validation / 1k test |
| Format | `{"prompt": str, "chosen": str, "rejected": str}` |

**Expected Outputs:**
| Output | Description |
|--------|-------------|
| `checkpoints/reward_model/` | Trained LoRA weights |
| `results/m1_reward_model/metrics.json` | Validation accuracy, loss curves |
| `results/m1_reward_model/reward_distribution.png` | Histogram of reward scores |
| `results/m1_reward_model/calibration.png` | Reward vs preference calibration |
| Inference function | `reward_model.score(prompt, response) → float` |

**Model Architecture:**
- Base: Qwen2-0.5B (`Qwen/Qwen2-0.5B`)
- Head: Linear layer on last token for scalar reward
- Training: LoRA (rank=16, alpha=32) for memory efficiency

**Training Configuration:**
- Batch size: 2 (gradient accumulation: 8)
- Learning rate: 2e-5 with cosine scheduler
- Epochs: 3
- Max sequence length: 512
- Device: MPS

**Metrics & Tracking:**
- Primary: Accuracy on held-out preference pairs
- Loss: Bradley-Terry pairwise ranking loss
- Reward hacking indicators:
  - Reward distribution statistics (mean, std, min, max)
  - Reward magnitude over training (detect unbounded growth)
  - Calibration plots (reward vs actual preference)
  - Length correlation (detect length bias)

**Deliverables:**
- `scripts/train_reward_model.py`
- `configs/reward_model.yaml`
- `src/reward_model/model.py`
- `src/data/hh_rlhf_loader.py`

**Success Criteria:**
- Validation accuracy > 65% on preference pairs
- Stable reward distribution (no unbounded growth)
- Low length-reward correlation (|r| < 0.3)

---

## Milestone 2: DPO (Direct Preference Optimization)

**Objective:** Align model directly from preferences without separate reward model

**Dataset:**
| Field | Value |
|-------|-------|
| Source | `Anthropic/hh-rlhf` (same as M1) |
| Size | 10,000 samples |
| Split | 8k train / 1k validation / 1k test |
| Format | `{"prompt": str, "chosen": str, "rejected": str}` |

**Expected Outputs:**
| Output | Description |
|--------|-------------|
| `checkpoints/dpo_model/` | Aligned LoRA weights |
| `results/m2_dpo/metrics.json` | Loss curves, KL divergence |
| `results/m2_dpo/win_rate.json` | Win rate vs base model |
| `results/m2_dpo/kl_trajectory.png` | KL over training |
| Aligned model | Can generate improved responses |

**Model:**
- Base: Qwen2-0.5B
- Training: LoRA (rank=16)
- Reference: Frozen copy of base model

**Key Hyperparameters:**
- Beta (KL penalty): 0.1
- Learning rate: 5e-6
- Batch size: 2

**Metrics:**
- Reward accuracy improvement
- KL divergence from reference
- Win rate on held-out comparisons

**Deliverables:**
- `scripts/train_dpo.py`
- `configs/dpo.yaml`

---

## Milestone 3: PPO (Proximal Policy Optimization)

**Objective:** Classic RLHF using reward model from M1

**Dataset:**
| Field | Value |
|-------|-------|
| Source | Prompts from `Anthropic/hh-rlhf` |
| Size | 10,000 prompts (no responses needed) |
| Format | `{"prompt": str}` |
| Reward | From M1 reward model |

**Expected Outputs:**
| Output | Description |
|--------|-------------|
| `checkpoints/ppo_model/` | PPO-trained LoRA weights |
| `results/m3_ppo/metrics.json` | Reward, KL, entropy curves |
| `results/m3_ppo/reward_trajectory.png` | Mean reward over training |
| `results/m3_ppo/kl_vs_reward.png` | KL-reward tradeoff plot |
| RLHF-aligned model | Policy optimized for M1 reward |

**Components:**
- Policy: Qwen2-0.5B (LoRA)
- Reward Model: From M1
- Reference: Frozen base model
- Value Head: Separate value function

**Key Hyperparameters:**
- PPO clip range: 0.2
- KL coefficient: 0.05
- Learning rate: 1e-5
- Mini-batch size: 4
- PPO epochs: 4

**Metrics:**
- Mean reward over training
- KL divergence from reference
- Policy entropy
- Value function loss

**Reward Hacking Monitoring:**
- Track reward vs KL tradeoff
- Monitor for reward model exploitation
- Evaluate on OOD prompts

**Deliverables:**
- `scripts/train_ppo.py`
- `configs/ppo.yaml`

---

## Milestone 4: GRPO (Group Relative Policy Optimization)

**Objective:** Reference-free policy optimization using group comparisons

**Dataset:**
| Field | Value |
|-------|-------|
| Source | Prompts from `Anthropic/hh-rlhf` |
| Size | 10,000 prompts |
| Format | `{"prompt": str}` |
| Reward | From M1 reward model (for ranking) |

**Expected Outputs:**
| Output | Description |
|--------|-------------|
| `checkpoints/grpo_model/` | GRPO-trained LoRA weights |
| `results/m4_grpo/metrics.json` | Ranking loss, reward curves |
| `results/m4_grpo/group_stats.json` | Per-group ranking statistics |
| `results/m4_grpo/diversity.json` | Response diversity metrics |
| GRPO-aligned model | Reference-free aligned policy |

**Method:**
- Sample multiple responses per prompt
- Rank by reward model
- Update policy toward higher-ranked responses
- No KL penalty to reference needed

**Key Hyperparameters:**
- Group size: 4 responses per prompt
- Temperature: 0.7 for generation
- Learning rate: 1e-5

**Metrics:**
- Group ranking accuracy
- Response quality improvement
- Generation diversity

**Deliverables:**
- `scripts/train_grpo.py`
- `configs/grpo.yaml`

---

## Milestone 5: Process Reward Model (PRM)

**Objective:** Train a reward model that scores individual reasoning steps, not just final answers

**Background:**
- Outcome Reward Model (ORM): Single reward for entire response
- Process Reward Model (PRM): Reward per reasoning step
- PRMs enable credit assignment to specific steps, improving reasoning quality

**Dataset:**
| Field | Value |
|-------|-------|
| Primary | `peiyi9979/Math-Shepherd` (HuggingFace) |
| Alternative | `openai/prm800k` or custom LLM-labeled |
| Size | 10,000 step-labeled examples |
| Format | `{"problem": str, "steps": [str], "labels": [0/1]}` |
| Domain | Math reasoning (GSM8K-style) |

**Expected Outputs:**
| Output | Description |
|--------|-------------|
| `checkpoints/prm_model/` | Trained PRM weights |
| `results/m5_prm/metrics.json` | Step accuracy, correlation metrics |
| `results/m5_prm/step_accuracy.png` | Per-step prediction accuracy |
| `results/m5_prm/error_detection.json` | Early error detection rates |
| Inference function | `prm.score_steps(problem, steps) → [float]` |

**Architecture:**
```
Input: [prompt] [step_1] [step_2] ... [step_n]
Output: reward_1, reward_2, ..., reward_n (per-step scores)
```

- Base: Qwen2-0.5B
- Head: Linear layer after each step delimiter token
- Training: Step-level cross-entropy or regression loss

**Step Labeling Strategies:**
1. **Human labels** (expensive, high quality)
2. **Outcome supervision** - backpropagate final correctness to steps
3. **LLM-as-judge** - use stronger model to label step correctness
4. **Monte Carlo estimation** - sample completions from each step, measure success rate

**Metrics:**
- Step-level accuracy (predict correct/incorrect steps)
- Correlation with final answer correctness
- Early error detection rate (catch mistakes at step N vs step N+k)

**Deliverables:**
- `src/reasoning/prm_model.py`
- `src/reasoning/step_tokenizer.py` - Parse reasoning into steps
- `scripts/train_prm.py`
- `configs/prm.yaml`

---

## Milestone 6: MCTS + LLM

**Objective:** Use Monte Carlo Tree Search to improve reasoning via search

**Dataset:**
| Field | Value |
|-------|-------|
| Evaluation | `gsm8k` (grade school math, 8.5K problems) |
| Alternative | `hendrycks/competition_math` (MATH dataset) |
| Size | 1,000 test problems for evaluation |
| Format | `{"question": str, "answer": str}` |
| Value Model | PRM from M5 or rollout-based |

**Expected Outputs:**
| Output | Description |
|--------|-------------|
| `src/reasoning/mcts.py` | MCTS implementation |
| `results/m6_mcts/solve_rates.json` | Accuracy: greedy vs MCTS |
| `results/m6_mcts/compute_scaling.png` | Accuracy vs compute budget |
| `results/m6_mcts/tree_stats.json` | Avg depth, branching, nodes explored |
| `results/m6_mcts/example_trees/` | Visualized search trees |
| Inference function | `mcts.solve(problem, budget) → solution` |

**Background:**
- Standard LLM: Greedy/sampling decode, no lookahead
- MCTS: Explore multiple reasoning paths, backpropagate success
- Combines LLM generation with tree search for better reasoning

**MCTS Components:**

1. **State:** Partial reasoning chain (prompt + steps so far)

2. **Actions:** Generate next reasoning step (sample from LLM)

3. **Value Function:**
   - Option A: Use PRM from M5 to score partial chains
   - Option B: Rollout to completion, check answer correctness
   - Option C: Hybrid (PRM + rollout)

4. **Policy:** LLM generation probabilities (can be uniform or guided)

**MCTS Algorithm:**
```
for iteration in range(N):
    1. SELECT: Traverse tree using UCB1 to find leaf
    2. EXPAND: Generate K candidate next steps from LLM
    3. EVALUATE: Score candidates using value function
    4. BACKPROPAGATE: Update node values up the tree

Return: Best path through tree
```

**Key Hyperparameters:**
| Parameter | Description | Values to Try |
|-----------|-------------|---------------|
| num_simulations | MCTS iterations per problem | 10, 50, 100 |
| branching_factor | Candidates per expansion | 3, 5, 10 |
| c_puct | Exploration constant in UCB | 1.0, 2.0, 4.0 |
| temperature | LLM sampling temp | 0.7, 1.0 |
| max_depth | Maximum reasoning steps | 5, 10, 15 |

**Evaluation:**
- Dataset: GSM8K (grade school math), MATH (competition math)
- Compare: Greedy decode vs beam search vs MCTS
- Metrics: Solve rate, steps to solution, compute cost

**Experiments:**
| Experiment | Description |
|------------|-------------|
| Greedy baseline | Standard autoregressive generation |
| MCTS + ORM | Search with outcome reward |
| MCTS + PRM | Search with process reward (M5) |
| Self-consistency | Sample N, majority vote |
| MCTS vs self-consistency | Compare at equal compute |

**Deliverables:**
- `src/reasoning/mcts.py` - Core MCTS implementation
- `src/reasoning/mcts_node.py` - Tree node structure
- `src/reasoning/value_functions.py` - PRM, rollout, hybrid
- `scripts/run_mcts.py`
- `configs/mcts.yaml`

---

## Experiment A: DPO vs PPO Comparison

**Objective:** Systematic comparison of DPO and PPO on identical data

**Setup:**
- Same dataset: Anthropic HH-RLHF 10k subset
- Same base model: Qwen2-0.5B with LoRA
- Same evaluation prompts and metrics

**Comparison Dimensions:**

| Metric | DPO | PPO |
|--------|-----|-----|
| Training stability | Track loss variance, gradient norms | Track reward variance, policy entropy |
| Compute cost | Wall-clock time, GPU memory, FLOPs | Wall-clock time, GPU memory, FLOPs |
| Sample efficiency | Samples to reach target performance | Samples to reach target performance |
| Final quality | Win rate on held-out pairs | Win rate on held-out pairs |

**Stability Metrics:**
- Loss/reward variance over training
- Number of training instabilities (NaN, spikes)
- Gradient norm statistics
- Convergence speed (steps to plateau)

**Compute Metrics:**
- Training time per epoch
- Peak memory usage
- Total training time to convergence

**Deliverables:**
- `scripts/compare_dpo_ppo.py` - Unified comparison script
- `results/comparison/` - Side-by-side results and plots

---

## Experiment B: DPO Beta (β) Ablation

**Objective:** Understand impact of β parameter on DPO training

**Background:** β controls the deviation penalty from reference policy
- Low β (0.01): Weak KL penalty, more aggressive updates
- Medium β (0.1): Balanced (default)
- High β (1.0): Strong KL penalty, conservative updates

**Ablation Grid:**
| β Value | Expected Behavior |
|---------|-------------------|
| 0.01 | Fast adaptation, risk of reward hacking, high KL |
| 0.1 | Balanced tradeoff (baseline) |
| 1.0 | Conservative, slow learning, low KL |

**Metrics per β:**
- Final reward accuracy
- KL divergence from reference
- Training loss curve
- Win rate on evaluation set
- Generation diversity (distinct n-grams)

**Analysis:**
- Plot reward vs KL tradeoff curves for each β
- Identify optimal β for this dataset/model
- Document stability differences

**Deliverables:**
- `scripts/ablate_beta.py`
- `configs/dpo_beta_ablation.yaml`
- `results/ablations/beta_sweep/`

---

## Experiment C: KL vs Reward Tracking (Reward Hacking Detection)

**Objective:** Monitor KL-reward tradeoff to detect reward hacking

**Background:** Reward hacking occurs when the policy exploits reward model weaknesses, achieving high reward while degrading actual quality.

**Tracking Setup:**
- Log KL divergence from reference at every N steps
- Log mean reward at every N steps
- Plot KL vs Reward curve over training

**Warning Signs:**
| Pattern | Interpretation |
|---------|----------------|
| Reward ↑, KL stable | Healthy learning |
| Reward ↑↑, KL ↑↑ | Potential reward hacking |
| Reward plateaus, KL ↑ | Definite exploitation |
| Reward spikes suddenly | Reward model vulnerability found |

**Automated Alerts:**
- KL exceeds threshold (e.g., > 10 nats)
- Reward growth rate exceeds KL-normalized threshold
- Reward variance spikes

**Deliverables:**
- `src/utils/kl_reward_tracker.py`
- Real-time plotting during training
- Automatic early stopping on hacking detection

---

## Experiment D: Frozen vs Unfrozen Value Head (PPO)

**Objective:** Compare value head training strategies in PPO

**Setup:**
| Configuration | Description |
|---------------|-------------|
| Frozen | Value head initialized, not updated during PPO |
| Unfrozen | Value head trained jointly with policy |
| Pretrained | Value head pretrained on reward model, then fine-tuned |

**Hypothesis:**
- Frozen: Faster, less compute, but may have stale value estimates
- Unfrozen: Better value estimates, but adds training complexity
- Pretrained: Best of both, but requires extra pretraining step

**Metrics:**
- Value function loss over training
- Policy gradient variance
- Final reward achieved
- Training stability (gradient norms)

**Deliverables:**
- `scripts/ablate_value_head.py`
- `configs/ppo_value_head_ablation.yaml`

---

## Experiment E: Reward Shaping

**Objective:** Implement and evaluate reward modifications

**Reward Components:**

1. **Base Reward:** From trained reward model (M1)

2. **Length Penalty:**
   ```
   R_length = -α * max(0, len(response) - target_len)
   ```
   - Penalize overly long responses
   - α = 0.01 per token over threshold
   - Target length: 256 tokens

3. **Format Bonus:**
   ```
   R_format = β * format_score(response)
   ```
   - Bonus for proper formatting (markdown, code blocks)
   - β = 0.1 for each format criterion met
   - Criteria: proper punctuation, paragraph breaks, code formatting

4. **Repetition Penalty:**
   ```
   R_repetition = -γ * repetition_ratio
   ```
   - Penalize repeated n-grams
   - γ = 0.5

**Final Shaped Reward:**
```
R_total = R_base + R_length + R_format + R_repetition
```

**Ablation:**
- Train with R_base only (baseline)
- Train with each shaping term individually
- Train with all shaping terms combined

**Deliverables:**
- `src/reward_model/reward_shaping.py`
- `configs/reward_shaping.yaml`
- Comparison results across configurations

---

## Experiment F: GRPO vs PPO Comparison

**Objective:** Compare GRPO and PPO on identical tasks

**Setup:**
- Same dataset: Anthropic HH-RLHF 10k subset
- Same base model: Qwen2-0.5B with LoRA
- Same reward model (from M1)

**Comparison Dimensions:**

| Metric | GRPO | PPO |
|--------|------|-----|
| Compute efficiency | No reference model forward pass | Requires reference + value head |
| Memory usage | Group samples in memory | Policy + value + reference |
| Samples per update | Group size × prompts | Batch size |
| Stability | Relative ranking (bounded) | Absolute rewards (unbounded) |

**Compute Metrics:**
- Training time per 1000 steps
- Peak GPU memory
- Samples processed per second
- Forward passes per update

**Stability Metrics:**
- Gradient norm variance
- Loss/reward variance
- Number of NaN/Inf occurrences
- Convergence consistency across seeds

**Deliverables:**
- `scripts/compare_grpo_ppo.py`
- `results/comparison/grpo_vs_ppo/`

---

## Experiment G: GRPO Group Size Ablation

**Objective:** Understand impact of group size on GRPO performance

**Ablation Grid:**
| Group Size | Trade-offs |
|------------|------------|
| 4 | Low compute, high variance in rankings |
| 8 | Balanced (baseline) |
| 16 | High compute, more stable rankings |

**Hypothesis:**
- Larger groups → more stable gradient estimates
- Larger groups → higher compute cost (more generations)
- Diminishing returns beyond certain size

**Metrics per Group Size:**
- Final reward achieved
- Training variance
- Compute cost (generations per update)
- Convergence speed
- Ranking quality (agreement with reward model)

**Deliverables:**
- `scripts/ablate_group_size.py`
- `configs/grpo_group_size_ablation.yaml`
- `results/ablations/group_size/`

---

## Experiment H: GRPO + DPO-style Regularization

**Objective:** Hybrid approach combining GRPO with DPO regularization

**Motivation:**
- GRPO: No explicit KL penalty, relies on relative ranking
- Risk: Policy may drift far from reference without constraint
- Solution: Add DPO-style KL regularization term

**Hybrid Loss:**
```
L_hybrid = L_grpo + λ * L_kl_regularization
```

Where:
- `L_grpo`: Standard GRPO ranking loss
- `L_kl_regularization`: KL(policy || reference) penalty
- `λ`: Regularization strength (ablate: 0.01, 0.1, 1.0)

**Comparison:**
| Method | KL Control | Reference Model |
|--------|------------|-----------------|
| Pure GRPO | None | Not needed |
| GRPO + KL | Explicit penalty | Required |
| DPO | Implicit in loss | Required |

**Metrics:**
- Final reward vs KL tradeoff
- Training stability
- Policy diversity
- OOD generalization

**Deliverables:**
- `scripts/train_grpo_regularized.py`
- `configs/grpo_dpo_hybrid.yaml`
- Comparison plots: pure GRPO vs hybrid vs DPO

---

## Experiment I: PRM vs ORM for RL Training

**Objective:** Compare process vs outcome supervision for policy training

**Setup:**
- Train policy with PPO using ORM (M1)
- Train policy with PPO using PRM (M5)
- Same base model, same dataset

**Hypothesis:**
- PRM provides denser signal → faster learning
- PRM catches errors early → better credit assignment
- ORM may lead to reward hacking on final answer format

**Metrics:**
- Learning speed (steps to target accuracy)
- Final solve rate
- Step-level error rate in final policy
- Robustness to distribution shift

**Deliverables:**
- `scripts/compare_prm_orm.py`
- `results/reasoning/prm_vs_orm/`

---

## Experiment J: DPO vs PPO vs GRPO Head-to-Head Evaluation

**Objective:** Direct comparison of all three alignment methods on identical held-out test data

**Test Set Requirements:**
| Field | Value |
|-------|-------|
| Source | `Anthropic/hh-rlhf` (same dataset) |
| Size | 100 randomly selected examples |
| Constraint | NOT in train (8k) or validation (1k) sets |
| Selection | Random seed 42 from test split (1k) |
| Format | `{"prompt": str}` (generate responses from each model) |

**Models to Compare:**
| Model | Checkpoint | Training Method |
|-------|------------|-----------------|
| Base | `Qwen/Qwen2-0.5B` | No alignment (baseline) |
| DPO | `checkpoints/dpo_model/best` | Direct Preference Optimization (M2) |
| PPO | `checkpoints/ppo_model/best` | Proximal Policy Optimization (M3) |
| GRPO | `checkpoints/grpo_model/best` | Group Relative Policy Optimization (M4) |

**Evaluation Metrics:**

1. **Reward Model Scores (M1):**
   - Mean reward per method
   - Reward distribution (min, max, std)
   - Win rate vs base model

2. **Pairwise Comparisons:**
   - DPO vs PPO win rate
   - DPO vs GRPO win rate
   - PPO vs GRPO win rate
   - Bradley-Terry rankings

3. **Response Quality Metrics:**
   - Average response length
   - Repetition ratio (n-gram diversity)
   - Format quality score

4. **LLM-as-Judge Evaluation:**
   - Judge model: `google/gemini-2.0-flash-001` via OpenRouter
   - Criteria: Helpfulness, Harmlessness, Honesty (HHH)
   - Score: 1-10 per criterion

**Expected Outputs:**
| Output | Description |
|--------|-------------|
| `results/experiment_j/comparison_results.json` | Full comparison data |
| `results/experiment_j/pairwise_matrix.json` | Win/loss/tie matrix |
| `results/experiment_j/reward_distributions.png` | Reward histograms per method |
| `results/experiment_j/rankings.json` | Bradley-Terry rankings |
| `results/experiment_j/judge_scores.json` | LLM judge evaluations |
| `results/experiment_j/sample_responses.json` | Example responses for inspection |

**Comparison Table Format:**
```
| Metric              | Base  | DPO   | PPO   | GRPO  |
|---------------------|-------|-------|-------|-------|
| Mean Reward         | X.XXX | X.XXX | X.XXX | X.XXX |
| Win Rate vs Base    | -     | XX%   | XX%   | XX%   |
| Judge Score (Avg)   | X.X   | X.X   | X.X   | X.X   |
| Response Length     | XXX   | XXX   | XXX   | XXX   |
| Repetition Ratio    | X.XX  | X.XX  | X.XX  | X.XX  |
```

**Statistical Analysis:**
- Bootstrap confidence intervals for win rates
- Paired t-tests for reward differences
- Effect size (Cohen's d) between methods

**Generation Parameters:**
| Parameter | Value |
|-----------|-------|
| Temperature | 0.7 |
| Max length | 256 tokens |
| Top-p | 0.9 |
| Do sample | True |

**Deliverables:**
- `scripts/run_benchmark_comparison.py` - Main comparison script
- `scripts/judge_responses.py` - LLM-as-judge evaluation
- `configs/benchmark_comparison.yaml` - Configuration

**Dependencies:**
- Requires M1 (reward model) for scoring
- Requires M2 (DPO), M3 (PPO), M4 (GRPO) trained models

---

## Infrastructure

### Logging & Visualization

All milestones must save comprehensive logs for post-training visualization.

| Log Type | Format | Frequency | Metrics |
|----------|--------|-----------|---------|
| Training metrics | JSON Lines | Every step | loss, lr, grad_norm |
| Evaluation metrics | JSON | Every eval | accuracy, reward stats |
| Checkpoints | PyTorch | Every N steps | model weights, optimizer |
| W&B / TensorBoard | Native | Real-time | All metrics |

**Per-Milestone Logging:**

| Milestone | Key Metrics to Log |
|-----------|-------------------|
| M1 (Reward) | loss, accuracy, reward_mean, reward_std, reward_min, reward_max, length_correlation |
| M2 (DPO) | loss, chosen_reward, rejected_reward, reward_margin, kl_divergence, win_rate |
| M3 (PPO) | policy_loss, value_loss, entropy, kl_div, mean_reward, reward_std, clip_fraction |
| M4 (GRPO) | ranking_loss, group_accuracy, reward_spread, generation_diversity |
| M5 (PRM) | step_loss, step_accuracy, early_detection_rate, correlation_with_outcome |
| M6 (MCTS) | solve_rate, avg_nodes, avg_depth, compute_per_problem, value_accuracy |

**Log File Structure:**
```
results/<milestone>/
├── logs/
│   ├── train_metrics.jsonl    # Step-by-step training logs
│   ├── eval_metrics.json      # Periodic evaluation results
│   └── config.yaml            # Training configuration snapshot
├── plots/
│   ├── loss_curve.png
│   ├── reward_trajectory.png
│   ├── kl_vs_reward.png       # (for PPO/DPO)
│   └── ...
└── checkpoints/
    ├── best_model/
    └── step_N/
```

**Visualization Script:**
```bash
# Generate all plots from logs
python scripts/visualize.py --results-dir results/m1_reward_model/
```

**Required Plots per Milestone:**

| Milestone | Required Visualizations |
|-----------|------------------------|
| M1 | Loss curve, accuracy curve, reward distribution histogram, calibration plot, length vs reward scatter |
| M2 | DPO loss, reward margin over time, KL trajectory, win rate progression |
| M3 | Policy/value loss, reward curve, KL vs reward tradeoff, entropy over time, clip fraction |
| M4 | Ranking loss, group accuracy, reward spread, diversity metrics |
| M5 | Step accuracy curve, per-position accuracy heatmap, error detection ROC |
| M6 | Solve rate vs compute budget, tree depth distribution, node efficiency |

---

### Evaluation Benchmarks

Final model quality measured using:
| Benchmark | Type | Metrics |
|-----------|------|---------|
| MT-Bench | Multi-turn judge | Score 1-10, per-category breakdown |
| AlpacaEval | Single-turn judge | Win rate vs reference (text-davinci-003) |

**Judge Model:**
| Setting | Value |
|---------|-------|
| Provider | OpenRouter API |
| Model | `google/gemini-2.0-flash-001` |

**Evaluation Script:**
```bash
python scripts/evaluate_benchmarks.py --model checkpoints/dpo_model --benchmarks mt-bench alpaca-eval
```

---

### Reproducibility

| Requirement | Implementation |
|-------------|----------------|
| Random seeds | Set `torch`, `numpy`, `random` seeds in config |
| Config logging | Save full config YAML with each run |
| Environment | Log Python version, package versions |
| Git commit | Log current commit hash |

**Config Template:**
```yaml
seed: 42
model:
  name: Qwen/Qwen2-0.5B
  # ...
training:
  # ...
# Full config saved to results/<run>/config.yaml
```

---

### Checkpoint Strategy

| Checkpoint | When | Contents |
|------------|------|----------|
| Best | Lowest validation loss | Model weights, optimizer, metrics |
| Last | End of training | Model weights, optimizer, metrics |

Storage: `checkpoints/<milestone>/best/` and `checkpoints/<milestone>/last/`

---

### Milestone Dependencies

```
M1 (Reward Model)
 ├── M2 (DPO) ─────────────────┐
 ├── M3 (PPO) ← uses M1 reward │
 └── M4 (GRPO) ← uses M1 reward│
                               │
M5 (PRM) ──────────────────────┤
 └── M6 (MCTS) ← uses M5 PRM   │
                               │
Experiments A-I ← as dependencies allow
```

**Parallel Execution Options:**
- M2 (DPO) can run independently (no reward model needed)
- M3, M4 require M1 completion
- M5, M6 are independent of M1-M4
- User decides execution order based on resources

---

### Memory Management (Mac MPS)

| Strategy | Implementation |
|----------|----------------|
| Gradient checkpointing | `model.gradient_checkpointing_enable()` |
| Small batch size | Batch size 1-2, gradient accumulation 8-16 |
| LoRA | Only train adapter weights (rank=16) |
| FP16/BF16 | Mixed precision where MPS supports |

**Training Config for Mac:**
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  fp16: false  # MPS compatibility
  bf16: false
  max_seq_length: 512  # Reduce if OOM
```

---

### Prompt Format

Using Alpaca format for all generation:

```
### Instruction:
{user_prompt}

### Response:
{model_response}
```

**Tokenization:**
- Add EOS token after response
- Mask instruction tokens in loss computation
- Handle multi-turn by concatenating instruction/response pairs

---

### Training Recovery

Auto-resume from last checkpoint on crash/restart:

```bash
# Automatically resumes if checkpoint exists
python scripts/train_reward_model.py --config configs/reward_model.yaml

# Force fresh start
python scripts/train_reward_model.py --config configs/reward_model.yaml --no-resume
```

---

### Data Preprocessing

Filter HH-RLHF by length only:

| Filter | Threshold |
|--------|-----------|
| Min length | 10 tokens |
| Max length | 512 tokens (prompt + response) |
| Remove | Empty, null, or truncated examples |

**Preprocessing Script:**
```bash
python scripts/preprocess_data.py --dataset anthropic/hh-rlhf --max-length 512 --output data/hh_rlhf_filtered/
```

---

### Reward Hacking Auto-Stop

Automatic training termination on reward hacking detection:

| Trigger | Threshold | Action |
|---------|-----------|--------|
| KL spike | KL > 15 nats | Stop training |
| Reward spike | Δreward > 3σ in 100 steps | Stop training |
| Reward plateau + KL rise | reward flat, KL ↑ 50% | Stop training |

**Implementation:**
```python
class RewardHackingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs, **kwargs):
        if logs.get("kl_divergence", 0) > self.kl_threshold:
            logger.warning(f"KL spike detected: {logs['kl_divergence']}")
            control.should_training_stop = True
```

**Output:** Save reason for early stop in `results/<run>/early_stop_reason.json`

---

### Code Structure

Modular architecture with shared utilities:

```
RLCode/
├── src/
│   ├── models/
│   │   ├── reward_model.py      # Reward model architecture
│   │   ├── prm_model.py         # Process reward model
│   │   └── value_head.py        # Value head for PPO
│   ├── trainers/
│   │   ├── reward_trainer.py    # Reward model training
│   │   ├── dpo_trainer.py       # DPO training loop
│   │   ├── ppo_trainer.py       # PPO training loop
│   │   └── grpo_trainer.py      # GRPO training loop
│   ├── data/
│   │   ├── hh_rlhf_loader.py    # HH-RLHF dataset
│   │   ├── math_shepherd.py     # Math-Shepherd for PRM
│   │   └── preprocessing.py     # Shared preprocessing
│   ├── reasoning/
│   │   ├── mcts.py              # MCTS implementation
│   │   ├── mcts_node.py         # Tree node structure
│   │   └── step_tokenizer.py    # Reasoning step parsing
│   ├── eval/
│   │   ├── judge.py             # OpenRouter judge API
│   │   ├── mt_bench.py          # MT-Bench evaluation
│   │   └── alpaca_eval.py       # AlpacaEval evaluation
│   └── utils/
│       ├── logger.py            # Unified logging
│       ├── metrics_tracker.py   # Metrics accumulation
│       ├── config.py            # Config loading/validation
│       └── callbacks.py         # Training callbacks
├── scripts/
│   ├── train_*.py               # Training scripts (call src/)
│   ├── evaluate_*.py            # Evaluation scripts
│   └── visualize.py             # Plot generation
├── configs/
│   └── *.yaml                   # All configurations
└── tests/
    └── smoke/                   # Smoke tests
```

---

### Testing Strategy

Smoke tests for quick sanity checks:

| Test | Purpose | Data |
|------|---------|------|
| `test_reward_model_forward.py` | Model loads, forward pass works | 1 sample |
| `test_data_loading.py` | Dataset loads correctly | 10 samples |
| `test_training_step.py` | One training step completes | 1 batch |
| `test_checkpoint_save.py` | Checkpoint saves/loads | Dummy model |

**Run Smoke Tests:**
```bash
pytest tests/smoke/ -v --timeout=60
```

---

### Training Verbosity

Progress bar with periodic key metrics:

```
Training: 100%|████████| 1000/1000 [10:32<00:00, 1.58it/s, loss=0.432, acc=0.71]
Step 100 | loss: 0.521 | accuracy: 0.68 | reward_mean: 1.23
Step 200 | loss: 0.487 | accuracy: 0.70 | reward_mean: 1.45
...
```

**Logging Levels:**
| Level | Content |
|-------|---------|
| INFO | Step metrics every 100 steps |
| DEBUG | Every step, full metrics |
| WARNING | OOM, gradient issues, early stop |
| ERROR | Training failures |

---

### Dependencies

**requirements.txt:**
```
torch>=2.0
transformers>=4.35
datasets
trl>=0.7
peft>=0.6
accelerate
bitsandbytes
wandb
numpy
pandas
matplotlib
scikit-learn
```

---

### Environment Setup

```bash
export OPENROUTER_API_KEY="your-key"  # For evaluation judging
```
