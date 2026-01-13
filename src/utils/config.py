"""Configuration loading and validation utilities."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2-0.5B"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05


@dataclass
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    scheduler: str = "cosine"
    epochs: int = 3
    max_seq_length: int = 512
    device: str = "mps"
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    dataset: str = "Anthropic/hh-rlhf"
    num_samples: int = 10000
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    min_length: int = 10
    max_length: int = 512


@dataclass
class DPOConfig:
    beta: float = 0.1  # KL penalty coefficient
    loss_type: str = "sigmoid"  # sigmoid or hinge


@dataclass
class PPOConfig:
    learning_rate: float = 1e-5
    kl_coef: float = 0.05  # KL penalty coefficient
    clip_range: float = 0.2  # PPO clip range
    clip_range_vf: float = 0.2  # Value function clip range
    vf_coef: float = 0.1  # Value function loss coefficient
    ppo_epochs: int = 4  # PPO epochs per batch
    mini_batch_size: int = 4
    target_kl: float = 6.0  # Target KL for early stopping
    init_kl_coef: float = 0.2  # Initial KL coefficient


@dataclass
class GRPOConfig:
    learning_rate: float = 1e-5
    group_size: int = 4  # Number of responses per prompt
    beta: float = 0.1  # Regularization coefficient (optional KL penalty)
    use_kl_penalty: bool = False  # Whether to use KL penalty to reference
    clip_range: float = 0.2  # Policy gradient clip range
    normalize_rewards: bool = True  # Normalize rewards within group


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class LoggingConfig:
    log_every_n_steps: int = 50
    eval_every_n_steps: int = 200
    save_every_n_steps: int = 500


@dataclass
class OutputConfig:
    checkpoint_dir: str = "checkpoints/reward_model"
    results_dir: str = "results/m1_reward_model"
    reward_model_path: str = "checkpoints/reward_model/best"


@dataclass
class Config:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw_config = yaml.safe_load(f)

    config = Config(
        seed=raw_config.get("seed", 42),
        model=ModelConfig(**raw_config.get("model", {})),
        training=TrainingConfig(**raw_config.get("training", {})),
        data=DataConfig(**raw_config.get("data", {})),
        dpo=DPOConfig(**raw_config.get("dpo", {})),
        ppo=PPOConfig(**raw_config.get("ppo", {})),
        grpo=GRPOConfig(**raw_config.get("grpo", {})),
        generation=GenerationConfig(**raw_config.get("generation", {})),
        logging=LoggingConfig(**raw_config.get("logging", {})),
        output=OutputConfig(**raw_config.get("output", {})),
    )

    return config


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "seed": config.seed,
        "model": {
            "name": config.model.name,
            "lora_rank": config.model.lora_rank,
            "lora_alpha": config.model.lora_alpha,
            "lora_target_modules": config.model.lora_target_modules,
            "lora_dropout": config.model.lora_dropout,
        },
        "training": {
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "learning_rate": config.training.learning_rate,
            "scheduler": config.training.scheduler,
            "epochs": config.training.epochs,
            "max_seq_length": config.training.max_seq_length,
            "device": config.training.device,
            "gradient_checkpointing": config.training.gradient_checkpointing,
            "warmup_ratio": config.training.warmup_ratio,
            "weight_decay": config.training.weight_decay,
            "max_grad_norm": config.training.max_grad_norm,
        },
        "data": {
            "dataset": config.data.dataset,
            "num_samples": config.data.num_samples,
            "train_split": config.data.train_split,
            "val_split": config.data.val_split,
            "test_split": config.data.test_split,
            "min_length": config.data.min_length,
            "max_length": config.data.max_length,
        },
        "dpo": {
            "beta": config.dpo.beta,
            "loss_type": config.dpo.loss_type,
        },
        "logging": {
            "log_every_n_steps": config.logging.log_every_n_steps,
            "eval_every_n_steps": config.logging.eval_every_n_steps,
            "save_every_n_steps": config.logging.save_every_n_steps,
        },
        "output": {
            "checkpoint_dir": config.output.checkpoint_dir,
            "results_dir": config.output.results_dir,
        },
    }

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
