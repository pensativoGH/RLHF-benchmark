#!/usr/bin/env python3
"""Evaluation script for PPO-trained model - compares against base model."""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hh_rlhf_loader import get_raw_data
from src.models.reward_model import RewardModel


def load_ppo_model(checkpoint_path: str, base_model_name: str, device: str):
    """Load the PPO-trained model (base + LoRA adapter)."""
    print(f"Loading PPO model from {checkpoint_path}...")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.to(device)
    model.eval()

    return model


def load_base_model(model_name: str, device: str):
    """Load the base model (no fine-tuning)."""
    print(f"Loading base model: {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = model.to(device)
    model.eval()

    return model


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 128, device: str = "mps"):
    """Generate a response from a model."""
    # Format prompt
    if prompt.startswith("Human:"):
        instruction = prompt[6:].strip()
    else:
        instruction = prompt.strip()

    formatted_prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate(
    ppo_checkpoint: str = "checkpoints/ppo_model/final",
    reward_model_path: str = "checkpoints/reward_model/best",
    base_model_name: str = "Qwen/Qwen2-0.5B",
    num_samples: int = 20,
    device: str = "mps",
):
    """Main evaluation function."""

    # Load tokenizer
    print(f"\nLoading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    ppo_model = load_ppo_model(ppo_checkpoint, base_model_name, device)
    base_model = load_base_model(base_model_name, device)

    # Load reward model
    print(f"Loading reward model from {reward_model_path}...")
    reward_model = RewardModel.load(
        checkpoint_path=reward_model_path,
        model_name=base_model_name,
        device=device,
    )
    reward_model.eval()

    # Get test prompts from HH-RLHF (use different seed for test set)
    print(f"\nLoading {num_samples} test prompts...")
    raw_data = get_raw_data(num_samples=num_samples * 2, seed=999, min_length=10)
    test_prompts = [item["prompt"] for item in raw_data[:num_samples]]

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION: PPO Model vs Base Model")
    print("=" * 70)

    results = []
    ppo_wins = 0
    base_wins = 0
    ties = 0

    ppo_rewards = []
    base_rewards = []

    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Sample {i+1}/{num_samples} ---")

        # Truncate prompt for display
        display_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
        print(f"Prompt: {display_prompt}")

        # Generate responses
        ppo_response = generate_response(ppo_model, tokenizer, prompt, device=device)
        base_response = generate_response(base_model, tokenizer, prompt, device=device)

        # Score with reward model
        ppo_reward = reward_model.score_batch(
            prompts=[prompt],
            responses=[ppo_response],
            tokenizer=tokenizer,
            max_length=512,
        )[0]

        base_reward = reward_model.score_batch(
            prompts=[prompt],
            responses=[base_response],
            tokenizer=tokenizer,
            max_length=512,
        )[0]

        ppo_rewards.append(ppo_reward)
        base_rewards.append(base_reward)

        # Determine winner
        if ppo_reward > base_reward + 0.01:  # Small margin for tie
            winner = "PPO"
            ppo_wins += 1
        elif base_reward > ppo_reward + 0.01:
            winner = "Base"
            base_wins += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"\nPPO Response (reward={ppo_reward:.4f}):")
        print(f"  {ppo_response[:300]}{'...' if len(ppo_response) > 300 else ''}")

        print(f"\nBase Response (reward={base_reward:.4f}):")
        print(f"  {base_response[:300]}{'...' if len(base_response) > 300 else ''}")

        print(f"\nWinner: {winner}")

        results.append({
            "prompt": prompt,
            "ppo_response": ppo_response,
            "base_response": base_response,
            "ppo_reward": float(ppo_reward),
            "base_reward": float(base_reward),
            "winner": winner,
        })

    # Summary statistics
    avg_ppo_reward = sum(ppo_rewards) / len(ppo_rewards)
    avg_base_reward = sum(base_rewards) / len(base_rewards)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total samples: {num_samples}")
    print(f"\nWin Rate:")
    print(f"  PPO wins:  {ppo_wins} ({100*ppo_wins/num_samples:.1f}%)")
    print(f"  Base wins: {base_wins} ({100*base_wins/num_samples:.1f}%)")
    print(f"  Ties:      {ties} ({100*ties/num_samples:.1f}%)")
    print(f"\nAverage Rewards:")
    print(f"  PPO model:  {avg_ppo_reward:.4f}")
    print(f"  Base model: {avg_base_reward:.4f}")
    print(f"  Improvement: {avg_ppo_reward - avg_base_reward:+.4f} ({100*(avg_ppo_reward - avg_base_reward)/abs(avg_base_reward):+.1f}%)")
    print("=" * 70)

    # Save results
    output_path = Path("results/m3_ppo/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_samples": num_samples,
        "ppo_wins": ppo_wins,
        "base_wins": base_wins,
        "ties": ties,
        "ppo_win_rate": ppo_wins / num_samples,
        "avg_ppo_reward": avg_ppo_reward,
        "avg_base_reward": avg_base_reward,
        "reward_improvement": avg_ppo_reward - avg_base_reward,
        "samples": results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO-trained model")
    parser.add_argument(
        "--ppo-checkpoint",
        type=str,
        default="checkpoints/ppo_model/final",
        help="Path to PPO model checkpoint",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default="checkpoints/reward_model/best",
        help="Path to reward model checkpoint",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Base model name",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use",
    )
    args = parser.parse_args()

    evaluate(
        ppo_checkpoint=args.ppo_checkpoint,
        reward_model_path=args.reward_model,
        base_model_name=args.base_model,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
