#!/usr/bin/env python3
"""Training script for reward model (Milestone 1)."""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hh_rlhf_loader import load_hh_rlhf
from src.models.reward_model import RewardModel, compute_pairwise_loss
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


def evaluate(
    model: RewardModel,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model on a dataset.

    Returns metrics including accuracy, reward statistics, and length correlation.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_chosen_rewards = []
    all_rejected_rewards = []
    all_chosen_lengths = []
    all_rejected_lengths = []

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            # Forward pass
            chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = model(rejected_input_ids, rejected_attention_mask)

            # Compute loss and accuracy
            loss, accuracy = compute_pairwise_loss(chosen_rewards, rejected_rewards)

            batch_size = chosen_input_ids.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (chosen_rewards > rejected_rewards).sum().item()
            total_samples += batch_size

            # Collect for statistics
            all_chosen_rewards.extend(chosen_rewards.cpu().tolist())
            all_rejected_rewards.extend(rejected_rewards.cpu().tolist())
            all_chosen_lengths.extend(batch["chosen_length"].tolist())
            all_rejected_lengths.extend(batch["rejected_length"].tolist())

    # Compute statistics
    chosen_rewards_arr = np.array(all_chosen_rewards)
    rejected_rewards_arr = np.array(all_rejected_rewards)
    all_rewards = np.concatenate([chosen_rewards_arr, rejected_rewards_arr])
    all_lengths = all_chosen_lengths + all_rejected_lengths

    # Length-reward correlation
    length_corr, _ = pearsonr(all_lengths, all_rewards.tolist())

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "chosen_reward_mean": float(np.mean(chosen_rewards_arr)),
        "chosen_reward_std": float(np.std(chosen_rewards_arr)),
        "rejected_reward_mean": float(np.mean(rejected_rewards_arr)),
        "rejected_reward_std": float(np.std(rejected_rewards_arr)),
        "reward_margin": float(np.mean(chosen_rewards_arr - rejected_rewards_arr)),
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "reward_min": float(np.min(all_rewards)),
        "reward_max": float(np.max(all_rewards)),
        "length_correlation": float(length_corr) if not np.isnan(length_corr) else 0.0,
    }

    model.train()
    return metrics


def train(config_path: str, resume: bool = True) -> None:
    """Main training function."""

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
    logger = MetricsLogger(str(logs_dir))

    # Load tokenizer
    print(f"Loading tokenizer: {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("Loading HH-RLHF dataset...")
    train_dataset, val_dataset, test_dataset = load_hh_rlhf(
        num_samples=config.data.num_samples,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        tokenizer=tokenizer,
        max_length=config.training.max_seq_length,
        min_length=config.data.min_length,
        seed=config.seed,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,  # MPS compatibility
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Initialize model
    print("Initializing reward model...")
    model = RewardModel(
        model_name=config.model.name,
        lora_rank=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_target_modules=config.model.lora_target_modules,
        lora_dropout=config.model.lora_dropout,
        device=config.training.device,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Setup scheduler
    total_steps = len(train_loader) * config.training.epochs // config.training.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    # Training state
    global_step = 0
    best_val_accuracy = 0.0
    accumulated_loss = 0.0

    print(f"\nStarting training...")
    print(f"  Total epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.training.batch_size * config.training.gradient_accumulation_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    for epoch in range(config.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.training.epochs}",
            leave=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            chosen_input_ids = batch["chosen_input_ids"].to(config.training.device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(config.training.device)
            rejected_input_ids = batch["rejected_input_ids"].to(config.training.device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(config.training.device)

            # Forward pass
            chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = model(rejected_input_ids, rejected_attention_mask)

            # Compute loss
            loss, accuracy = compute_pairwise_loss(chosen_rewards, rejected_rewards)
            loss = loss / config.training.gradient_accumulation_steps

            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()

            # Track epoch metrics
            batch_size = chosen_input_ids.shape[0]
            epoch_loss += loss.item() * config.training.gradient_accumulation_steps * batch_size
            epoch_correct += (chosen_rewards > rejected_rewards).sum().item()
            epoch_samples += batch_size

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

                # Update progress bar
                current_lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix({
                    "loss": f"{accumulated_loss:.4f}",
                    "acc": f"{accuracy.item():.2%}",
                    "lr": f"{current_lr:.2e}",
                })

                # Log training metrics
                if global_step % config.logging.log_every_n_steps == 0:
                    logger.log(
                        step=global_step,
                        metrics={
                            "loss": accumulated_loss,
                            "accuracy": accuracy.item(),
                            "learning_rate": current_lr,
                            "grad_norm": grad_norm.item(),
                        },
                        phase="train",
                        epoch=epoch + 1,
                    )

                accumulated_loss = 0.0

                # Evaluate
                if global_step % config.logging.eval_every_n_steps == 0:
                    val_metrics = evaluate(model, val_loader, config.training.device)
                    logger.log_eval(global_step, val_metrics, epoch=epoch + 1)

                    # Save best model
                    if val_metrics["accuracy"] > best_val_accuracy:
                        best_val_accuracy = val_metrics["accuracy"]
                        model.save(str(checkpoint_dir / "best"))
                        print(f"\n  New best accuracy: {best_val_accuracy:.2%}")

                # Save checkpoint
                if global_step % config.logging.save_every_n_steps == 0:
                    model.save(str(checkpoint_dir / f"step_{global_step}"))

        # End of epoch summary
        epoch_accuracy = epoch_correct / epoch_samples
        epoch_avg_loss = epoch_loss / epoch_samples
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {epoch_avg_loss:.4f}")
        print(f"  Train Accuracy: {epoch_accuracy:.2%}")

        # Full validation at end of epoch
        val_metrics = evaluate(model, val_loader, config.training.device)
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.2%}")
        print(f"  Length Correlation: {val_metrics['length_correlation']:.3f}")

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            model.save(str(checkpoint_dir / "best"))
            print(f"  New best accuracy: {best_val_accuracy:.2%}")

    # Save final model
    model.save(str(checkpoint_dir / "last"))

    # Final test evaluation
    print("\nFinal Test Evaluation:")
    test_metrics = evaluate(model, test_loader, config.training.device)
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"  Length Correlation: {test_metrics['length_correlation']:.3f}")

    # Save final metrics
    final_metrics = {
        "best_val_accuracy": best_val_accuracy,
        "test": test_metrics,
        "config": {
            "model": config.model.name,
            "lora_rank": config.model.lora_rank,
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "num_samples": config.data.num_samples,
        },
    }
    logger.save_final_metrics(final_metrics)

    # Check success criteria
    print("\n" + "=" * 50)
    print("SUCCESS CRITERIA CHECK:")
    print(f"  Validation Accuracy > 65%: {best_val_accuracy:.2%} {'PASS' if best_val_accuracy > 0.65 else 'FAIL'}")
    print(f"  Length Correlation |r| < 0.3: {abs(test_metrics['length_correlation']):.3f} {'PASS' if abs(test_metrics['length_correlation']) < 0.3 else 'FAIL'}")
    print(f"  Reward Range: [{test_metrics['reward_min']:.2f}, {test_metrics['reward_max']:.2f}]")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reward_model.yaml",
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
