#!/usr/bin/env python3
"""Training script for PPO (Proximal Policy Optimization) - Milestone 3.

Uses TRL's new PPOTrainer API (v0.26+).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOConfig as TRLPPOConfig
from trl import PPOTrainer

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


def prepare_prompts_dataset(raw_data: list) -> Dataset:
    """Convert raw data to HuggingFace Dataset with prompts.

    The new TRL PPOTrainer expects a 'input_ids' or text column.
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
        })

    return Dataset.from_list(formatted_data)


class RewardModelWrapper(nn.Module):
    """Wrapper to make our RewardModel compatible with TRL's PPOTrainer.

    TRL expects a model with:
    - base_model_prefix attribute pointing to the backbone
    - score() method that takes hidden states and returns per-token scores (batch, seq_len)
    """

    def __init__(self, reward_model: RewardModel):
        super().__init__()
        self.reward_model = reward_model
        # Expose the base model for TRL's get_reward function
        self.model = reward_model.model.base_model  # The underlying transformer
        self.base_model_prefix = "model"
        # Expose config for TRL compatibility
        self.config = reward_model.base_model.config
        # Keep the reward head
        self.reward_head = reward_model.reward_head

    def score(self, hidden_states):
        """Score function expected by TRL's get_reward.

        Returns per-token scores of shape (batch, seq_len).
        """
        # hidden_states: [batch, seq_len, hidden_size]
        scores = self.reward_head(hidden_states)  # [batch, seq_len, 1]
        return scores.squeeze(-1)  # [batch, seq_len]

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        rewards = self.reward_model.forward(input_ids, attention_mask)
        return rewards


class ValueModel(nn.Module):
    """Value model for PPO - predicts value of states.

    Uses the same architecture as the reward model but with a separate head.
    The structure must match TRL's PolicyAndValueWrapper expectations.
    """

    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        # The "model" attribute is what base_model_prefix points to
        self.model = base_model  # TRL expects getattr(self, base_model_prefix)
        self.base_model_prefix = "model"  # For TRL compatibility
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.value_head.weight)

    def score(self, hidden_states):
        """Score function expected by TRL's get_reward.

        TRL expects per-token scores of shape (batch, seq_len).
        """
        # hidden_states: [batch, seq_len, hidden_size]
        scores = self.value_head(hidden_states)  # [batch, seq_len, 1]
        return scores.squeeze(-1)  # [batch, seq_len]

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        return self.score(hidden_states)


def train(config_path: str) -> None:
    """Main PPO training function."""

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

    # Load raw data and prepare prompts
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

    # Prepare datasets
    train_dataset = prepare_prompts_dataset(train_data)
    eval_dataset = prepare_prompts_dataset(eval_data)

    # Tokenize datasets and remove prompt column
    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=config.data.max_length // 2,  # Leave room for response
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["prompt"])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["prompt"])

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        target_modules=config.model.lora_target_modules,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load policy model
    print(f"Loading policy model: {config.model.name}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # MPS compatibility
    )

    # Load reward model from M1
    reward_model_path = config.output.reward_model_path
    print(f"Loading reward model from: {reward_model_path}")
    reward_model = RewardModel.load(
        checkpoint_path=reward_model_path,
        model_name=config.model.name,
        device=config.training.device,
    )
    reward_model.eval()
    wrapped_reward_model = RewardModelWrapper(reward_model)

    # Create value model
    print("Creating value model...")
    value_base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    # Enable gradient checkpointing for memory efficiency
    if config.training.gradient_checkpointing:
        value_base_model.gradient_checkpointing_enable()
    hidden_size = value_base_model.config.hidden_size
    value_model = ValueModel(value_base_model, hidden_size)

    # Calculate total episodes
    total_episodes = len(train_dataset) * config.training.epochs

    # Configure PPO training
    ppo_config = TRLPPOConfig(
        output_dir=str(checkpoint_dir),
        learning_rate=config.ppo.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_ppo_epochs=config.ppo.ppo_epochs,
        cliprange=config.ppo.clip_range,
        cliprange_value=config.ppo.clip_range_vf,
        vf_coef=config.ppo.vf_coef,
        kl_coef=config.ppo.kl_coef,
        seed=config.seed,
        total_episodes=total_episodes,
        response_length=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        report_to="none",  # Disable wandb
        logging_steps=config.logging.log_every_n_steps,
        save_steps=config.logging.save_every_n_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )

    print(f"\nPPO Training Configuration:")
    print(f"  Learning rate: {config.ppo.learning_rate}")
    print(f"  KL coefficient: {config.ppo.kl_coef}")
    print(f"  Clip range: {config.ppo.clip_range}")
    print(f"  PPO epochs: {config.ppo.ppo_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Total episodes: {total_episodes}")

    # Initialize PPO trainer
    print("\nInitializing PPO trainer...")
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=None,  # Will create copy automatically when using PEFT
        reward_model=wrapped_reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    # Train
    print("\nStarting PPO training...")
    ppo_trainer.train()

    # Save final model
    print("\nSaving final model...")
    ppo_trainer.save_model(str(checkpoint_dir / "final"))

    # Get training metrics from trainer state
    if hasattr(ppo_trainer, 'state') and hasattr(ppo_trainer.state, 'log_history'):
        training_history = ppo_trainer.state.log_history
    else:
        training_history = []

    # Save training history
    history_path = logs_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2, default=str)

    # Compute final metrics
    final_metrics = {
        "config": {
            "model": config.model.name,
            "lora_rank": config.model.lora_rank,
            "learning_rate": config.ppo.learning_rate,
            "kl_coef": config.ppo.kl_coef,
            "clip_range": config.ppo.clip_range,
            "epochs": config.training.epochs,
            "num_samples": config.data.num_samples,
        },
    }

    # Save final metrics
    metrics_logger.save_final_metrics(final_metrics)

    print("\n" + "=" * 50)
    print("PPO TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Model saved to: {checkpoint_dir}")
    print(f"  Results saved to: {results_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Train PPO model (Milestone 3)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
