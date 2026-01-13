"""Reward model architecture with LoRA for preference learning."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


class RewardModel(nn.Module):
    """Reward model for scoring responses.

    Architecture:
    - Base: Qwen2-0.5B (or other causal LM)
    - LoRA adapters for efficient fine-tuning
    - Linear reward head on last token hidden state
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.05,
        device: str = "mps",
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name

        # Load base model
        print(f"Loading base model: {model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # MPS compatibility
            trust_remote_code=True,
        )

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        # Apply LoRA
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()

        # Get hidden size from config
        hidden_size = self.base_model.config.hidden_size

        # Reward head: linear layer on last token hidden state
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.reward_head.weight)

        # Move to device
        self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to compute reward scores.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            rewards: Scalar rewards [batch_size]
        """
        # Get hidden states from the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

        # Find the last non-padding token position for each sequence
        # attention_mask is 1 for real tokens, 0 for padding
        sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
        batch_size = input_ids.shape[0]

        # Gather hidden states at the last token position
        last_hidden = hidden_states[
            torch.arange(batch_size, device=input_ids.device),
            sequence_lengths,
        ]  # [batch_size, hidden_size]

        # Compute reward
        rewards = self.reward_head(last_hidden).squeeze(-1)  # [batch_size]

        return rewards

    def score(
        self,
        prompt: str,
        response: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> float:
        """Score a single prompt-response pair.

        Args:
            prompt: The instruction/prompt
            response: The model response
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length

        Returns:
            Scalar reward score
        """
        from src.data.hh_rlhf_loader import format_for_reward_model

        self.eval()
        with torch.no_grad():
            text = format_for_reward_model(prompt, response)
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            reward = self.forward(input_ids, attention_mask)

        return reward.item()

    def score_batch(
        self,
        prompts: List[str],
        responses: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> List[float]:
        """Score a batch of prompt-response pairs.

        Args:
            prompts: List of prompts
            responses: List of responses
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length

        Returns:
            List of scalar reward scores
        """
        from src.data.hh_rlhf_loader import format_for_reward_model

        self.eval()
        with torch.no_grad():
            texts = [
                format_for_reward_model(p, r) for p, r in zip(prompts, responses)
            ]
            encoding = tokenizer(
                texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            rewards = self.forward(input_ids, attention_mask)

        return rewards.tolist()

    def save(self, save_path: str) -> None:
        """Save the LoRA adapters and reward head."""
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save LoRA adapters
        self.model.save_pretrained(save_path)

        # Save reward head
        torch.save(
            self.reward_head.state_dict(),
            os.path.join(save_path, "reward_head.pt"),
        )
        print(f"Model saved to {save_path}")

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        model_name: str = "Qwen/Qwen2-0.5B",
        device: str = "mps",
    ) -> "RewardModel":
        """Load a saved reward model.

        Args:
            checkpoint_path: Path to saved checkpoint
            model_name: Base model name
            device: Device to load to

        Returns:
            Loaded RewardModel
        """
        import os

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, checkpoint_path)

        # Create reward model instance
        hidden_size = base_model.config.hidden_size
        reward_model = cls.__new__(cls)
        nn.Module.__init__(reward_model)

        reward_model.device = device
        reward_model.model_name = model_name
        reward_model.base_model = base_model
        reward_model.model = model

        # Create and load reward head
        reward_model.reward_head = nn.Linear(hidden_size, 1, bias=False)
        reward_head_path = os.path.join(checkpoint_path, "reward_head.pt")
        if os.path.exists(reward_head_path):
            reward_model.reward_head.load_state_dict(torch.load(reward_head_path))

        reward_model.to(device)
        print(f"Model loaded from {checkpoint_path}")

        return reward_model


def compute_pairwise_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Bradley-Terry pairwise ranking loss.

    Loss = -log(sigmoid(r_chosen - r_rejected))

    Args:
        chosen_rewards: Rewards for chosen responses [batch_size]
        rejected_rewards: Rewards for rejected responses [batch_size]

    Returns:
        Tuple of (loss, accuracy)
    """
    # Compute margin
    margin = chosen_rewards - rejected_rewards

    # Bradley-Terry loss
    loss = -torch.nn.functional.logsigmoid(margin).mean()

    # Accuracy: fraction where chosen > rejected
    accuracy = (margin > 0).float().mean()

    return loss, accuracy
