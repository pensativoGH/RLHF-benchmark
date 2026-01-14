#!/usr/bin/env python3
"""Evaluate new GRPO model on existing test prompts."""

import json
import numpy as np
import torch
from pathlib import Path
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.reward_model import RewardModel


def main():
    device = "mps"

    # Load existing results
    print("Loading existing results...")
    with open("results/experiment_j/full_results.json") as f:
        existing_results = json.load(f)

    prompts = [r["prompt"] for r in existing_results]
    print(f"Loaded {len(prompts)} prompts")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load new GRPO model
    print("Loading new GRPO model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, "checkpoints/grpo_model/final")
    model = model.to(device)
    model.eval()

    # Load reward model
    print("Loading reward model...")
    reward_model = RewardModel.load(
        checkpoint_path="checkpoints/reward_model/best",
        model_name="Qwen/Qwen2-0.5B",
        device=device,
    )

    # Generate and score responses
    print("\nGenerating new GRPO responses...")
    new_grpo_rewards = []
    new_grpo_responses = []

    for i, prompt in enumerate(tqdm(prompts, desc="New GRPO")):
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        new_grpo_responses.append(response)

        # Score
        reward = reward_model.score(prompt, response, tokenizer)
        new_grpo_rewards.append(reward)

    # Compute stats
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    # Get old rewards
    old_grpo_rewards = [r["grpo_reward"] for r in existing_results]
    base_rewards = [r["base_reward"] for r in existing_results]
    dpo_rewards = [r["dpo_reward"] for r in existing_results]
    ppo_rewards = [r["ppo_reward"] for r in existing_results]

    print("\n### Mean Rewards ###")
    print(f"{'Model':<15} {'Mean':>10} {'Std':>10} {'Median':>10}")
    print("-" * 50)
    print(f"{'Base':<15} {np.mean(base_rewards):>10.4f} {np.std(base_rewards):>10.4f} {np.median(base_rewards):>10.4f}")
    print(f"{'DPO':<15} {np.mean(dpo_rewards):>10.4f} {np.std(dpo_rewards):>10.4f} {np.median(dpo_rewards):>10.4f}")
    print(f"{'PPO':<15} {np.mean(ppo_rewards):>10.4f} {np.std(ppo_rewards):>10.4f} {np.median(ppo_rewards):>10.4f}")
    print(f"{'Old GRPO':<15} {np.mean(old_grpo_rewards):>10.4f} {np.std(old_grpo_rewards):>10.4f} {np.median(old_grpo_rewards):>10.4f}")
    print(f"{'New GRPO':<15} {np.mean(new_grpo_rewards):>10.4f} {np.std(new_grpo_rewards):>10.4f} {np.median(new_grpo_rewards):>10.4f}")

    # Win rates
    def win_rate(rewards_a, rewards_b, threshold=0.01):
        wins = sum(1 for a, b in zip(rewards_a, rewards_b) if a > b + threshold)
        losses = sum(1 for a, b in zip(rewards_a, rewards_b) if b > a + threshold)
        ties = len(rewards_a) - wins - losses
        return wins, losses, ties

    print("\n### Win Rates ###")

    # New GRPO vs Base
    w, l, t = win_rate(new_grpo_rewards, base_rewards)
    print(f"New GRPO vs Base: {w} wins, {l} losses, {t} ties ({w}% win rate)")

    # New GRPO vs DPO
    w, l, t = win_rate(new_grpo_rewards, dpo_rewards)
    print(f"New GRPO vs DPO:  {w} wins, {l} losses, {t} ties ({w}% win rate)")

    # New GRPO vs PPO
    w, l, t = win_rate(new_grpo_rewards, ppo_rewards)
    print(f"New GRPO vs PPO:  {w} wins, {l} losses, {t} ties ({w}% win rate)")

    # New GRPO vs Old GRPO
    w, l, t = win_rate(new_grpo_rewards, old_grpo_rewards)
    print(f"New GRPO vs Old:  {w} wins, {l} losses, {t} ties ({w}% win rate)")

    # Compare with previous benchmarks
    print("\n### Comparison with Previous ###")
    print(f"Old GRPO vs Base: 49% win rate")
    print(f"Old GRPO vs PPO:  31% win rate")

    # Save results
    results = {
        "new_grpo_rewards": new_grpo_rewards,
        "new_grpo_mean": float(np.mean(new_grpo_rewards)),
        "new_grpo_std": float(np.std(new_grpo_rewards)),
        "comparisons": {
            "new_grpo_vs_base": win_rate(new_grpo_rewards, base_rewards),
            "new_grpo_vs_dpo": win_rate(new_grpo_rewards, dpo_rewards),
            "new_grpo_vs_ppo": win_rate(new_grpo_rewards, ppo_rewards),
            "new_grpo_vs_old_grpo": win_rate(new_grpo_rewards, old_grpo_rewards),
        }
    }

    with open("results/experiment_j/new_grpo_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to results/experiment_j/new_grpo_comparison.json")


if __name__ == "__main__":
    main()
