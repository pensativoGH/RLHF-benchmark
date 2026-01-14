#!/usr/bin/env python3
"""Training script for GRPO (Group Relative Policy Optimization) - Milestone 4.

GRPO is a reference-free policy optimization method that:
1. Samples multiple responses (group) per prompt
2. Ranks responses using a reward model
3. Updates policy toward higher-ranked responses within each group

Key difference from PPO: No KL penalty to reference model needed.
The relative ranking within groups provides implicit regularization.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hh_rlhf_loader import get_raw_data
from src.models.reward_model import RewardModel
from src.utils.config import load_config, save_config
from src.utils.logger import MetricsLogger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def format_prompt(prompt: str) -> str:
    """Format prompt in Alpaca style."""
    if prompt.startswith("Human:"):
        instruction = prompt[6:].strip()
    else:
        instruction = prompt.strip()
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_responses(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> Tuple[List[List[str]], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    """Generate multiple responses per prompt.

    Note: Uses sequential generation for MPS stability. num_return_sequences
    causes numerical issues with MPS backend.

    Returns:
        responses: List of response strings per prompt [batch, group_size]
        response_ids: List of token ID tensors per response
        response_log_probs: List of log probability tensors per response
    """
    model.eval()

    all_responses = []
    all_response_ids = []
    all_log_probs = []

    with torch.no_grad():
        for prompt in prompts:
            # Tokenize prompt once
            prompt_encoding = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            prompt_len = prompt_encoding["input_ids"].shape[1]

            prompt_responses = []
            prompt_response_ids = []
            prompt_log_probs = []

            # Sequential generation (MPS-stable)
            for _ in range(group_size):
                outputs = model.generate(
                    **prompt_encoding,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                generated_ids = outputs.sequences[0, prompt_len:]
                response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Compute log probabilities
                log_probs = []
                for i, score in enumerate(outputs.scores):
                    if i < len(generated_ids):
                        token_id = generated_ids[i].item()
                        log_prob = F.log_softmax(score[0], dim=-1)[token_id]
                        log_probs.append(log_prob.item())

                prompt_responses.append(response_text)
                prompt_response_ids.append(generated_ids)
                prompt_log_probs.append(torch.tensor(log_probs, device=device))

            all_responses.append(prompt_responses)
            all_response_ids.append(prompt_response_ids)
            all_log_probs.append(prompt_log_probs)

    return all_responses, all_response_ids, all_log_probs


def compute_rewards(
    reward_model: RewardModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[List[str]],
) -> List[List[float]]:
    """Compute rewards for all prompt-response pairs.

    Returns:
        rewards: [batch, group_size] list of reward scores
    """
    all_rewards = []

    for prompt, group_responses in zip(prompts, responses):
        group_rewards = []
        for response in group_responses:
            reward = reward_model.score(prompt, response, tokenizer)
            group_rewards.append(reward)
        all_rewards.append(group_rewards)

    return all_rewards


def normalize_rewards(
    rewards: List[List[float]],
    epsilon: float = 1e-8,
) -> List[List[float]]:
    """Normalize rewards within each group (zero mean, unit variance).

    This is the key to GRPO - relative rewards within groups.
    Uses rank-based advantages when variance is too low to avoid zero gradients.
    """
    normalized = []

    for group_rewards in rewards:
        group_arr = np.array(group_rewards)
        mean = group_arr.mean()
        std = group_arr.std()

        if std < epsilon:
            # Use rank-based advantages instead of zeros to preserve gradient signal
            n = len(group_arr)
            if n > 1:
                # Get ranks (0 to n-1) and map to [-1, 1] range
                ranks = np.argsort(np.argsort(group_arr)).astype(float)
                normalized_group = ((2.0 * ranks / (n - 1)) - 1.0).tolist()
            else:
                normalized_group = [0.0]
        else:
            normalized_group = ((group_arr - mean) / (std + epsilon)).tolist()

        normalized.append(normalized_group)

    return normalized


def compute_kl_penalty(
    model: nn.Module,
    ref_model: nn.Module,
    prompts: List[str],
    responses: List[List[str]],
    tokenizer: AutoTokenizer,
    device: str,
) -> float:
    """Compute KL divergence from reference model.

    Returns the mean KL divergence per token across all responses.
    Memory-optimized: processes one response at a time with explicit cleanup.
    """
    total_kl = 0.0
    num_tokens = 0

    model.eval()
    with torch.no_grad():
        for prompt, group_responses in zip(prompts, responses):
            for response in group_responses:
                full_text = prompt + response
                encoding = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,  # Reduced for memory
                ).to(device)

                prompt_enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                ).to(device)
                prompt_len = prompt_enc["input_ids"].shape[1]

                # Get logits from policy model
                policy_outputs = model(**encoding, return_dict=True)
                policy_logits = policy_outputs.logits[0, prompt_len - 1:-1, :].clone()
                del policy_outputs

                # Get logits from reference model
                ref_outputs = ref_model(**encoding, return_dict=True)
                ref_logits = ref_outputs.logits[0, prompt_len - 1:-1, :].clone()
                del ref_outputs

                if len(policy_logits) == 0:
                    del policy_logits, ref_logits, encoding, prompt_enc
                    continue

                # Compute KL divergence token by token to save memory
                # Using approximation: KL ≈ (log_p - log_q) for sampled tokens
                # This is more memory efficient than full softmax
                response_ids = encoding["input_ids"][0, prompt_len:]
                min_len = min(len(policy_logits), len(response_ids))

                if min_len > 0:
                    policy_lp = F.log_softmax(policy_logits[:min_len], dim=-1)
                    ref_lp = F.log_softmax(ref_logits[:min_len], dim=-1)

                    # Get log probs for actual tokens only (memory efficient)
                    token_ids = response_ids[:min_len]
                    policy_token_lp = torch.gather(policy_lp, 1, token_ids.unsqueeze(1)).squeeze(1)
                    ref_token_lp = torch.gather(ref_lp, 1, token_ids.unsqueeze(1)).squeeze(1)

                    # Approximate KL using sampled tokens
                    kl = (policy_token_lp - ref_token_lp).clamp(min=0)  # Non-negative KL
                    total_kl += kl.sum().item()
                    num_tokens += min_len

                    del policy_lp, ref_lp, policy_token_lp, ref_token_lp, kl

                del policy_logits, ref_logits, encoding, prompt_enc

                # Clear MPS cache periodically
                if device == "mps" and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()

    model.train()
    mean_kl = total_kl / max(1, num_tokens)
    return mean_kl


def adjust_kl_coef(current_kl: float, target_kl: float, kl_coef: float) -> float:
    """Adaptively adjust KL coefficient based on current vs target KL.

    Mirrors PPO's adaptive KL penalty strategy:
    - If KL > 1.5 * target: increase penalty (policy drifting too fast)
    - If KL < 0.5 * target: decrease penalty (can be more aggressive)
    """
    if current_kl > target_kl * 1.5:
        # KL too high, increase penalty to constrain policy
        return min(kl_coef * 1.5, 1.0)
    elif current_kl < target_kl * 0.5:
        # KL too low, decrease penalty to allow more exploration
        return max(kl_coef / 1.5, 0.01)
    return kl_coef


def compute_grpo_loss(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[List[str]],
    advantages: List[List[float]],
    old_log_probs: List[List[torch.Tensor]],
    clip_range: float,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute GRPO policy gradient loss.

    Uses clipped surrogate objective similar to PPO but with group-normalized advantages.

    Loss = -E[min(r*A, clip(r, 1-eps, 1+eps)*A)]
    where r = pi(a|s) / pi_old(a|s), A = normalized group advantage
    """
    model.train()

    total_loss = 0.0
    num_tokens = 0
    clip_fractions = []
    approx_kls = []

    for prompt, group_responses, group_advantages, group_old_log_probs in zip(
        prompts, responses, advantages, old_log_probs
    ):
        for response, advantage, old_log_prob in zip(
            group_responses, group_advantages, group_old_log_probs
        ):
            if len(old_log_prob) == 0:
                continue

            # Tokenize full sequence
            full_text = prompt + response
            encoding = tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            prompt_encoding = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            prompt_len = prompt_encoding["input_ids"].shape[1]

            # Forward pass to get current log probs
            outputs = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                return_dict=True,
            )

            # Get logits for response tokens
            logits = outputs.logits[0, prompt_len - 1:-1, :]  # [response_len, vocab]
            response_ids = encoding["input_ids"][0, prompt_len:]  # [response_len]

            # Compute current log probs
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, 1, response_ids.unsqueeze(1)
            ).squeeze(1)

            # Match lengths (generation might differ from tokenization)
            min_len = min(len(token_log_probs), len(old_log_prob))
            if min_len == 0:
                continue

            current_lp = token_log_probs[:min_len]
            old_lp = old_log_prob[:min_len].to(device)

            # Compute ratio and clipped objective
            ratio = torch.exp(current_lp - old_lp)
            advantage_tensor = torch.tensor(advantage, device=device)

            # Clipped surrogate loss
            surr1 = ratio * advantage_tensor
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage_tensor

            # Take the minimum (pessimistic bound)
            loss = -torch.min(surr1, surr2).mean()

            total_loss += loss * min_len
            num_tokens += min_len

            # Track metrics
            clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean().item()
            clip_fractions.append(clip_fraction)

            approx_kl = (old_lp - current_lp).mean().item()
            approx_kls.append(approx_kl)

    if num_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    avg_loss = total_loss / num_tokens

    metrics = {
        "policy_loss": avg_loss.item(),
        "clip_fraction": np.mean(clip_fractions) if clip_fractions else 0.0,
        "approx_kl": np.mean(approx_kls) if approx_kls else 0.0,
    }

    return avg_loss, metrics


def compute_diversity_metrics(responses: List[List[str]]) -> Dict[str, float]:
    """Compute diversity metrics for generated responses."""
    all_responses_flat = [r for group in responses for r in group]

    if not all_responses_flat:
        return {"unique_ratio": 0.0, "avg_length": 0.0}

    # Unique response ratio
    unique_ratio = len(set(all_responses_flat)) / len(all_responses_flat)

    # Average response length
    avg_length = np.mean([len(r.split()) for r in all_responses_flat])

    # Within-group diversity (average pairwise distinctness)
    group_diversities = []
    for group in responses:
        if len(group) > 1:
            unique_in_group = len(set(group))
            group_diversities.append(unique_in_group / len(group))

    avg_group_diversity = np.mean(group_diversities) if group_diversities else 0.0

    return {
        "unique_ratio": unique_ratio,
        "avg_response_length": avg_length,
        "avg_group_diversity": avg_group_diversity,
    }


def compute_ranking_accuracy(rewards: List[List[float]]) -> float:
    """Compute how well the ranking correlates with reward ordering.

    For each group, check if highest-reward response has highest normalized advantage.
    """
    correct = 0
    total = 0

    for group_rewards in rewards:
        if len(group_rewards) < 2:
            continue

        # Find index of best response
        best_idx = np.argmax(group_rewards)

        # In normalized rewards, highest reward should have positive advantage
        normalized = normalize_rewards([group_rewards])[0]
        predicted_best = np.argmax(normalized)

        if best_idx == predicted_best:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_policy(
    model: nn.Module,
    reward_model: RewardModel,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    max_samples: int = 50,
) -> Dict[str, float]:
    """Evaluate policy on held-out data."""
    model.eval()

    prompts = [format_prompt(item["prompt"]) for item in eval_data[:max_samples]]

    # Generate responses
    responses, _, _ = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )

    # Compute rewards
    rewards = compute_rewards(reward_model, tokenizer, prompts, responses)

    # Compute metrics
    all_rewards_flat = [r for group in rewards for r in group]
    diversity_metrics = compute_diversity_metrics(responses)
    ranking_accuracy = compute_ranking_accuracy(rewards)

    metrics = {
        "mean_reward": np.mean(all_rewards_flat),
        "std_reward": np.std(all_rewards_flat),
        "max_reward": np.max(all_rewards_flat),
        "min_reward": np.min(all_rewards_flat),
        "ranking_accuracy": ranking_accuracy,
        **diversity_metrics,
    }

    return metrics


def train(config_path: str) -> None:
    """Main GRPO training function."""

    # Load config
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")

    # Set seed
    set_seed(config.seed)
    print(f"Set random seed: {config.seed}")

    # Create output directories
    results_dir = Path(config.output.results_dir)
    logs_dir = results_dir / "logs"
    checkpoint_dir = Path(config.output.checkpoint_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    save_config(config, str(logs_dir / "config.yaml"))

    # Initialize logger
    metrics_logger = MetricsLogger(str(logs_dir))

    # Load tokenizer
    print(f"Loading tokenizer: {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load raw data
    print("Loading HH-RLHF dataset...")
    raw_data = get_raw_data(
        num_samples=config.data.num_samples,
        seed=config.seed,
        min_length=config.data.min_length,
    )

    # Split data
    n_train = int(len(raw_data) * config.data.train_split)
    train_data = raw_data[:n_train]
    eval_data = raw_data[n_train:]

    print(f"Split sizes - Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Load policy model
    print(f"Loading policy model: {config.model.name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # MPS compatibility
    )

    # Enable gradient checkpointing
    if config.training.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        target_modules=config.model.lora_target_modules,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model = model.to(config.training.device)

    # Load reward model from M1
    reward_model_path = config.output.reward_model_path
    print(f"Loading reward model from: {reward_model_path}")
    reward_model = RewardModel.load(
        checkpoint_path=reward_model_path,
        model_name=config.model.name,
        device=config.training.device,
    )
    reward_model.eval()

    # Load reference model for KL penalty (if enabled)
    ref_model = None
    if config.grpo.use_kl_penalty:
        print("Loading reference model for KL penalty...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        ref_model = ref_model.to(config.training.device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("  Reference model loaded (frozen)")

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.grpo.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Calculate total steps
    steps_per_epoch = len(train_data) // config.training.batch_size
    total_steps = steps_per_epoch * config.training.epochs // config.training.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    # Setup scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))

    print(f"\nGRPO Training Configuration:")
    print(f"  Group size: {config.grpo.group_size}")
    print(f"  Learning rate: {config.grpo.learning_rate}")
    print(f"  Clip range: {config.grpo.clip_range}")
    print(f"  Normalize rewards: {config.grpo.normalize_rewards}")
    print(f"  Use KL penalty: {config.grpo.use_kl_penalty}")
    if config.grpo.use_kl_penalty:
        print(f"  Target KL: {config.grpo.target_kl}")
        print(f"  Initial KL coef: {config.grpo.init_kl_coef}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # Training state
    global_step = 0
    best_mean_reward = float("-inf")
    training_history = []
    group_stats_history = []
    diversity_history = []

    # Initialize adaptive KL coefficient
    kl_coef = config.grpo.init_kl_coef if config.grpo.use_kl_penalty else 0.0

    print("\nStarting GRPO training...")

    for epoch in range(config.training.epochs):
        # Shuffle training data
        random.shuffle(train_data)

        epoch_metrics = defaultdict(list)

        progress_bar = tqdm(
            range(0, len(train_data), config.training.batch_size),
            desc=f"Epoch {epoch + 1}/{config.training.epochs}",
        )

        for batch_idx, start_idx in enumerate(progress_bar):
            # Get batch
            batch_data = train_data[start_idx:start_idx + config.training.batch_size]
            prompts = [format_prompt(item["prompt"]) for item in batch_data]

            # Generate responses for each prompt
            responses, response_ids, old_log_probs = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                group_size=config.grpo.group_size,
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
                device=config.training.device,
            )

            # Compute rewards
            rewards = compute_rewards(reward_model, tokenizer, prompts, responses)

            # Normalize rewards within groups (GRPO key insight)
            if config.grpo.normalize_rewards:
                advantages = normalize_rewards(rewards)
            else:
                advantages = rewards

            # Compute GRPO loss
            loss, step_metrics = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                responses=responses,
                advantages=advantages,
                old_log_probs=old_log_probs,
                clip_range=config.grpo.clip_range,
                device=config.training.device,
            )

            # Compute and add KL penalty if enabled (only at gradient accumulation boundaries for speed)
            mean_kl = 0.0
            kl_penalty = 0.0
            compute_kl_this_step = (
                config.grpo.use_kl_penalty
                and ref_model is not None
                and (batch_idx + 1) % config.training.gradient_accumulation_steps == 0
            )
            if compute_kl_this_step:
                # Use subset of responses (first 2 per prompt) for speed
                subset_responses = [[r[:2] if len(r) > 2 else r for r in responses][0]]
                mean_kl = compute_kl_penalty(
                    model=model,
                    ref_model=ref_model,
                    prompts=prompts[:1],  # Just first prompt
                    responses=subset_responses,
                    tokenizer=tokenizer,
                    device=config.training.device,
                )
                kl_penalty = kl_coef * mean_kl
                # Add KL penalty to loss (scaled by accumulation steps since computed once per acc step)
                loss = loss + kl_penalty * config.training.gradient_accumulation_steps
                step_metrics["kl_divergence"] = mean_kl
                step_metrics["kl_coef"] = kl_coef
                step_metrics["kl_penalty"] = kl_penalty

            # Scale loss for gradient accumulation
            scaled_loss = loss / config.training.gradient_accumulation_steps
            scaled_loss.backward()

            # Collect metrics
            all_rewards_flat = [r for group in rewards for r in group]
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["mean_reward"].append(np.mean(all_rewards_flat))
            epoch_metrics["reward_std"].append(np.std(all_rewards_flat))

            for key, value in step_metrics.items():
                epoch_metrics[key].append(value)

            # Gradient accumulation step
            if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.max_grad_norm,
                )

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # Scheduler step (after warmup)
                if global_step >= warmup_steps:
                    scheduler.step()

                global_step += 1

                # Adaptive KL coefficient adjustment
                if config.grpo.use_kl_penalty:
                    recent_kl = epoch_metrics.get("kl_divergence", [0])[-config.training.gradient_accumulation_steps:]
                    if recent_kl:
                        avg_recent_kl = np.mean(recent_kl)
                        kl_coef = adjust_kl_coef(avg_recent_kl, config.grpo.target_kl, kl_coef)

                # Update progress bar
                current_lr = optimizer.param_groups[0]["lr"]
                postfix = {
                    "loss": f"{np.mean(epoch_metrics['loss'][-config.training.gradient_accumulation_steps:]):.4f}",
                    "reward": f"{np.mean(epoch_metrics['mean_reward'][-config.training.gradient_accumulation_steps:]):.4f}",
                    "lr": f"{current_lr:.2e}",
                }
                if config.grpo.use_kl_penalty:
                    postfix["kl"] = f"{np.mean(epoch_metrics.get('kl_divergence', [0])[-config.training.gradient_accumulation_steps:]):.3f}"
                progress_bar.set_postfix(postfix)

                # Log metrics
                if global_step % config.logging.log_every_n_steps == 0:
                    log_metrics = {
                        "loss": np.mean(epoch_metrics["loss"][-config.training.gradient_accumulation_steps:]),
                        "mean_reward": np.mean(epoch_metrics["mean_reward"][-config.training.gradient_accumulation_steps:]),
                        "reward_std": np.mean(epoch_metrics["reward_std"][-config.training.gradient_accumulation_steps:]),
                        "clip_fraction": np.mean(epoch_metrics.get("clip_fraction", [0])[-config.training.gradient_accumulation_steps:]),
                        "approx_kl": np.mean(epoch_metrics.get("approx_kl", [0])[-config.training.gradient_accumulation_steps:]),
                        "learning_rate": current_lr,
                        "grad_norm": grad_norm.item(),
                    }
                    # Add KL penalty metrics if enabled
                    if config.grpo.use_kl_penalty:
                        log_metrics["kl_divergence"] = np.mean(epoch_metrics.get("kl_divergence", [0])[-config.training.gradient_accumulation_steps:])
                        log_metrics["kl_coef"] = kl_coef
                        log_metrics["kl_penalty"] = np.mean(epoch_metrics.get("kl_penalty", [0])[-config.training.gradient_accumulation_steps:])
                    metrics_logger.log(global_step, log_metrics, phase="train", epoch=epoch + 1)

                    training_history.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        **log_metrics,
                    })

                # Evaluate
                if global_step % config.logging.eval_every_n_steps == 0:
                    print("\nRunning evaluation...")
                    eval_metrics = evaluate_policy(
                        model=model,
                        reward_model=reward_model,
                        tokenizer=tokenizer,
                        eval_data=eval_data,
                        group_size=config.grpo.group_size,
                        max_new_tokens=config.generation.max_new_tokens,
                        temperature=config.generation.temperature,
                        top_p=config.generation.top_p,
                        device=config.training.device,
                        max_samples=50,
                    )
                    metrics_logger.log_eval(global_step, eval_metrics, epoch=epoch + 1)

                    # Track diversity
                    diversity_history.append({
                        "step": global_step,
                        "unique_ratio": eval_metrics["unique_ratio"],
                        "avg_group_diversity": eval_metrics["avg_group_diversity"],
                        "avg_response_length": eval_metrics["avg_response_length"],
                    })

                    # Track group stats
                    group_stats_history.append({
                        "step": global_step,
                        "ranking_accuracy": eval_metrics["ranking_accuracy"],
                        "mean_reward": eval_metrics["mean_reward"],
                        "std_reward": eval_metrics["std_reward"],
                    })

                    # Save best model
                    if eval_metrics["mean_reward"] > best_mean_reward:
                        best_mean_reward = eval_metrics["mean_reward"]
                        model.save_pretrained(str(checkpoint_dir / "best"))
                        tokenizer.save_pretrained(str(checkpoint_dir / "best"))
                        print(f"  New best mean reward: {best_mean_reward:.4f}")

                # Save checkpoint
                if global_step % config.logging.save_every_n_steps == 0:
                    model.save_pretrained(str(checkpoint_dir / f"step_{global_step}"))

        # End of epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Mean Loss: {np.mean(epoch_metrics['loss']):.4f}")
        print(f"  Mean Reward: {np.mean(epoch_metrics['mean_reward']):.4f}")
        print(f"  Reward Std: {np.mean(epoch_metrics['reward_std']):.4f}")

    # Save final model
    print("\nSaving final model...")
    model.save_pretrained(str(checkpoint_dir / "final"))
    tokenizer.save_pretrained(str(checkpoint_dir / "final"))

    # Final evaluation
    print("\nRunning final evaluation...")
    final_eval_metrics = evaluate_policy(
        model=model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        eval_data=eval_data,
        group_size=config.grpo.group_size,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        device=config.training.device,
        max_samples=100,
    )

    # Save training history
    history_path = logs_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    # Save group stats
    group_stats_path = results_dir / "group_stats.json"
    with open(group_stats_path, "w") as f:
        json.dump(group_stats_history, f, indent=2)

    # Save diversity metrics
    diversity_path = results_dir / "diversity.json"
    with open(diversity_path, "w") as f:
        json.dump(diversity_history, f, indent=2)

    # Collect final metrics
    final_metrics = {
        "best_mean_reward": best_mean_reward,
        "final_eval": final_eval_metrics,
        "config": {
            "model": config.model.name,
            "lora_rank": config.model.lora_rank,
            "group_size": config.grpo.group_size,
            "learning_rate": config.grpo.learning_rate,
            "clip_range": config.grpo.clip_range,
            "normalize_rewards": config.grpo.normalize_rewards,
            "epochs": config.training.epochs,
            "num_samples": config.data.num_samples,
        },
    }

    # Save final metrics
    metrics_logger.save_final_metrics(final_metrics)

    print("\n" + "=" * 50)
    print("GRPO TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Best mean reward: {best_mean_reward:.4f}")
    print(f"  Final mean reward: {final_eval_metrics['mean_reward']:.4f}")
    print(f"  Final ranking accuracy: {final_eval_metrics['ranking_accuracy']:.2%}")
    print(f"  Final unique ratio: {final_eval_metrics['unique_ratio']:.2%}")
    print(f"  Model saved to: {checkpoint_dir}")
    print(f"  Results saved to: {results_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Train GRPO model (Milestone 4)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
