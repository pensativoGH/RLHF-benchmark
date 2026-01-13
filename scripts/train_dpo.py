#!/usr/bin/env python3
"""Training script for DPO (Direct Preference Optimization) - Milestone 2."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig as TRLDPOConfig
from trl import DPOTrainer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hh_rlhf_loader import get_raw_data
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


def prepare_dataset(raw_data: list, tokenizer: AutoTokenizer) -> Dataset:
    """Convert raw data to HuggingFace Dataset for DPOTrainer.

    DPOTrainer expects: {"prompt": str, "chosen": str, "rejected": str}
    """

    def format_prompt(prompt: str) -> str:
        """Format prompt in Alpaca style."""
        if prompt.startswith("Human:"):
            instruction = prompt[6:].strip()
        else:
            instruction = prompt.strip()
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    formatted_data = []
    for item in raw_data:
        formatted_data.append({
            "prompt": format_prompt(item["prompt"]),
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        })

    return Dataset.from_list(formatted_data)


def compute_win_rate(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    device: str,
    max_samples: int = 100,
) -> dict:
    """Compute win rate of trained model vs reference model.

    For each prompt, generate responses from both models and compare using
    implicit reward (log probability ratio).
    """
    model.eval()
    ref_model.eval()

    wins = 0
    ties = 0
    losses = 0
    total = min(len(eval_dataset), max_samples)

    kl_divergences = []
    reward_margins = []

    with torch.no_grad():
        for i in range(total):
            item = eval_dataset[i]
            prompt = item["prompt"]
            chosen = item["chosen"]

            # Tokenize prompt + chosen response
            full_text = prompt + chosen
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(device)

            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(device)
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # Get log probs from both models
            policy_outputs = model(**inputs)
            ref_outputs = ref_model(**inputs)

            # Compute log probs for response tokens only
            policy_logits = policy_outputs.logits[:, prompt_len - 1:-1, :]
            ref_logits = ref_outputs.logits[:, prompt_len - 1:-1, :]
            response_ids = inputs["input_ids"][:, prompt_len:]

            # Log softmax
            policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)

            # Gather log probs for actual tokens
            policy_token_log_probs = torch.gather(
                policy_log_probs, 2, response_ids.unsqueeze(-1)
            ).squeeze(-1)
            ref_token_log_probs = torch.gather(
                ref_log_probs, 2, response_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Sum over tokens
            policy_score = policy_token_log_probs.sum().item()
            ref_score = ref_token_log_probs.sum().item()

            # Implicit reward is the log prob difference
            reward_margin = policy_score - ref_score
            reward_margins.append(reward_margin)

            # KL divergence approximation (per-token average)
            num_tokens = response_ids.shape[1]
            if num_tokens > 0:
                kl = (policy_token_log_probs - ref_token_log_probs).mean().item()
                kl_divergences.append(kl)

            # Count wins (higher log prob = better)
            if policy_score > ref_score + 0.1:
                wins += 1
            elif policy_score < ref_score - 0.1:
                losses += 1
            else:
                ties += 1

    metrics = {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "tie_rate": ties / total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "total": total,
        "avg_reward_margin": float(np.mean(reward_margins)),
        "avg_kl_divergence": float(np.mean(kl_divergences)) if kl_divergences else 0.0,
    }

    model.train()
    return metrics


class DPOMetricsCallback:
    """Custom callback to extract DPO training metrics."""

    def __init__(self, logger: MetricsLogger):
        self.logger = logger
        self.metrics_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            metrics = {
                k: v for k, v in logs.items()
                if isinstance(v, (int, float)) and not k.startswith("_")
            }
            if metrics:
                self.metrics_history.append({"step": step, **metrics})
                # Log key metrics
                key_metrics = {}
                if "loss" in logs:
                    key_metrics["loss"] = logs["loss"]
                if "rewards/chosen" in logs:
                    key_metrics["chosen_reward"] = logs["rewards/chosen"]
                if "rewards/rejected" in logs:
                    key_metrics["rejected_reward"] = logs["rewards/rejected"]
                if "rewards/margins" in logs:
                    key_metrics["reward_margin"] = logs["rewards/margins"]
                if "logps/chosen" in logs:
                    key_metrics["chosen_logps"] = logs["logps/chosen"]
                if "logps/rejected" in logs:
                    key_metrics["rejected_logps"] = logs["logps/rejected"]
                if key_metrics:
                    self.logger.log(step, key_metrics, phase="train")


def train(config_path: str, resume: bool = True) -> None:
    """Main DPO training function."""

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
    tokenizer.padding_side = "left"  # Required for DPO

    # Load raw data and split
    print("Loading HH-RLHF dataset...")
    raw_data = get_raw_data(
        num_samples=config.data.num_samples,
        seed=config.seed,
        min_length=config.data.min_length,
    )

    # Split data
    n_train = int(len(raw_data) * config.data.train_split)
    n_val = int(len(raw_data) * config.data.val_split)

    train_data = raw_data[:n_train]
    val_data = raw_data[n_train:n_train + n_val]
    test_data = raw_data[n_train + n_val:]

    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    eval_dataset = prepare_dataset(val_data, tokenizer)
    test_dataset = prepare_dataset(test_data, tokenizer)

    # Load model
    print(f"Loading model: {config.model.name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # MPS compatibility
    )

    # Enable gradient checkpointing
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        target_modules=config.model.lora_target_modules,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load reference model (frozen, no LoRA)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    # Calculate total steps
    total_steps = (
        len(train_dataset)
        * config.training.epochs
        // (config.training.batch_size * config.training.gradient_accumulation_steps)
    )
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    # Configure DPO training
    training_args = TRLDPOConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        warmup_steps=warmup_steps,
        lr_scheduler_type=config.training.scheduler,
        logging_steps=config.logging.log_every_n_steps,
        eval_strategy="steps",
        eval_steps=config.logging.eval_every_n_steps,
        save_strategy="steps",
        save_steps=config.logging.save_every_n_steps,
        save_total_limit=3,
        bf16=False,  # MPS compatibility
        fp16=False,
        remove_unused_columns=False,
        beta=config.dpo.beta,
        loss_type=config.dpo.loss_type,
        max_length=config.training.max_seq_length,
        max_prompt_length=config.training.max_seq_length // 2,
        report_to="none",  # Disable wandb, etc.
    )

    print(f"\nDPO Training Configuration:")
    print(f"  Beta (KL penalty): {config.dpo.beta}")
    print(f"  Loss type: {config.dpo.loss_type}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.training.batch_size * config.training.gradient_accumulation_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting DPO training...")
    train_result = trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(checkpoint_dir / "final"))
    model.save_pretrained(str(checkpoint_dir / "final"))
    tokenizer.save_pretrained(str(checkpoint_dir / "final"))

    # Extract training metrics from trainer state
    train_metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
    }

    # Evaluate on test set
    print("\nEvaluating on test set...")

    # Move models to device for evaluation
    device = config.training.device
    model = model.to(device)
    ref_model = ref_model.to(device)

    # Compute win rate
    print("Computing win rate...")
    win_rate_metrics = compute_win_rate(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        eval_dataset=test_dataset,
        device=device,
        max_samples=min(100, len(test_dataset)),
    )

    print(f"\nWin Rate Results:")
    print(f"  Win rate: {win_rate_metrics['win_rate']:.2%}")
    print(f"  Loss rate: {win_rate_metrics['loss_rate']:.2%}")
    print(f"  Tie rate: {win_rate_metrics['tie_rate']:.2%}")
    print(f"  Average reward margin: {win_rate_metrics['avg_reward_margin']:.4f}")
    print(f"  Average KL divergence: {win_rate_metrics['avg_kl_divergence']:.4f}")

    # Save win rate metrics
    win_rate_path = results_dir / "win_rate.json"
    with open(win_rate_path, "w") as f:
        json.dump(win_rate_metrics, f, indent=2)
    print(f"Win rate metrics saved to {win_rate_path}")

    # Run evaluation on eval set
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()

    # Collect all metrics
    final_metrics = {
        "train": train_metrics,
        "eval": eval_results,
        "win_rate": win_rate_metrics,
        "config": {
            "model": config.model.name,
            "lora_rank": config.model.lora_rank,
            "beta": config.dpo.beta,
            "loss_type": config.dpo.loss_type,
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "num_samples": config.data.num_samples,
        },
    }

    # Save final metrics
    metrics_logger.save_final_metrics(final_metrics)

    # Extract and save training history for plotting
    training_history = []
    for log in trainer.state.log_history:
        if "loss" in log:
            training_history.append({
                "step": log.get("step", 0),
                "loss": log.get("loss"),
                "learning_rate": log.get("learning_rate"),
                "epoch": log.get("epoch"),
                "rewards/chosen": log.get("rewards/chosen"),
                "rewards/rejected": log.get("rewards/rejected"),
                "rewards/margins": log.get("rewards/margins"),
            })

    history_path = logs_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 50)
    print("DPO TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Final train loss: {train_metrics['train_loss']:.4f}")
    print(f"  Win rate vs base: {win_rate_metrics['win_rate']:.2%}")
    print(f"  Average KL divergence: {win_rate_metrics['avg_kl_divergence']:.4f}")
    print(f"  Model saved to: {checkpoint_dir / 'final'}")
    print(f"  Results saved to: {results_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Train DPO model (Milestone 2)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force fresh start (ignore existing checkpoints)",
    )
    args = parser.parse_args()

    train(args.config, resume=not args.no_resume)


if __name__ == "__main__":
    main()
