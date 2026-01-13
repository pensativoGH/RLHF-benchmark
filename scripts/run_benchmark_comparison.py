#!/usr/bin/env python3
"""
Experiment J: DPO vs PPO vs GRPO Head-to-Head Evaluation

Compares all three alignment methods on 100 held-out test examples.
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.reward_model import RewardModel


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_test_data(config: dict) -> list:
    """Load and sample test data from HH-RLHF."""
    print("Loading HH-RLHF dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")

    # Filter by length
    min_len = config['data']['min_length']
    max_len = config['data']['max_length']

    filtered_data = []
    for item in dataset:
        # Extract prompt from chosen (everything before last Assistant response)
        chosen = item['chosen']
        if 'Human:' in chosen and 'Assistant:' in chosen:
            # Find the last Human: message as the prompt
            parts = chosen.split('Human:')
            if len(parts) >= 2:
                # Get everything up to the last Assistant response
                last_human_idx = chosen.rfind('Human:')
                last_assistant_idx = chosen.rfind('Assistant:')
                if last_assistant_idx > last_human_idx:
                    prompt = chosen[:last_assistant_idx].strip()
                    if min_len <= len(prompt.split()) <= max_len * 2:
                        filtered_data.append({'prompt': prompt})

    print(f"Filtered to {len(filtered_data)} examples")

    # Sample test examples
    num_samples = config['data']['num_test_samples']
    set_seed(config['seed'])
    sampled_data = random.sample(filtered_data, min(num_samples, len(filtered_data)))

    print(f"Sampled {len(sampled_data)} test examples")
    return sampled_data


def load_base_model(model_name: str, device: str):
    """Load the base model and tokenizer."""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    return model, tokenizer


def load_aligned_model(base_model_name: str, adapter_path: str, device: str, tokenizer):
    """Load a model with LoRA adapter."""
    print(f"Loading aligned model from: {adapter_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to(device)
    model.eval()

    return model


def generate_response(model, tokenizer, prompt: str, config: dict, device: str) -> str:
    """Generate a response for a given prompt."""
    gen_config = config['generation']

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config['max_new_tokens'],
            temperature=gen_config['temperature'],
            top_p=gen_config['top_p'],
            do_sample=gen_config['do_sample'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return response.strip()


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    """Compute n-gram repetition ratio (lower is better)."""
    words = text.split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if len(ngrams) == 0:
        return 0.0

    unique_ngrams = set(ngrams)
    return 1 - (len(unique_ngrams) / len(ngrams))


def compute_pairwise_comparison(reward_a: float, reward_b: float, threshold: float = 0.01) -> str:
    """Determine winner based on reward scores."""
    if reward_a > reward_b + threshold:
        return "A"
    elif reward_b > reward_a + threshold:
        return "B"
    else:
        return "Tie"


def run_comparison(config: dict):
    """Run the full comparison experiment."""
    device = config['device']
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_data = load_test_data(config)

    # Load base model and tokenizer
    base_model, tokenizer = load_base_model(config['models']['base'], device)

    # Load reward model
    print("Loading reward model...")
    reward_model = RewardModel.load(
        checkpoint_path=config['models']['reward_model'],
        model_name=config['models']['base'],
        device=device,
    )

    # Initialize results storage
    all_results = []
    model_stats = {
        'base': {'rewards': [], 'lengths': [], 'repetitions': []},
        'dpo': {'rewards': [], 'lengths': [], 'repetitions': []},
        'ppo': {'rewards': [], 'lengths': [], 'repetitions': []},
        'grpo': {'rewards': [], 'lengths': [], 'repetitions': []},
    }

    # Pairwise comparison counters
    pairwise = {
        'dpo_vs_base': {'dpo': 0, 'base': 0, 'tie': 0},
        'ppo_vs_base': {'ppo': 0, 'base': 0, 'tie': 0},
        'grpo_vs_base': {'grpo': 0, 'base': 0, 'tie': 0},
        'dpo_vs_ppo': {'dpo': 0, 'ppo': 0, 'tie': 0},
        'dpo_vs_grpo': {'dpo': 0, 'grpo': 0, 'tie': 0},
        'ppo_vs_grpo': {'ppo': 0, 'grpo': 0, 'tie': 0},
    }

    # Generate responses from base model first
    print("\n=== Generating Base Model Responses ===")
    base_responses = []
    for item in tqdm(test_data, desc="Base model"):
        response = generate_response(base_model, tokenizer, item['prompt'], config, device)
        base_responses.append(response)

    # Score base responses
    print("Scoring base responses...")
    base_rewards = []
    for prompt, response in tqdm(zip([d['prompt'] for d in test_data], base_responses),
                                   total=len(test_data), desc="Scoring base"):
        reward = reward_model.score(prompt, response, tokenizer)
        base_rewards.append(reward)
        model_stats['base']['rewards'].append(reward)
        model_stats['base']['lengths'].append(len(response.split()))
        model_stats['base']['repetitions'].append(compute_repetition_ratio(response))

    # Free base model memory before loading aligned models
    del base_model
    torch.mps.empty_cache() if device == 'mps' else torch.cuda.empty_cache()

    # Process each aligned model
    aligned_models = ['dpo', 'ppo', 'grpo']
    aligned_responses = {}
    aligned_rewards = {}

    for model_name in aligned_models:
        print(f"\n=== Processing {model_name.upper()} Model ===")

        # Load model
        model = load_aligned_model(
            config['models']['base'],
            config['models'][model_name],
            device,
            tokenizer
        )

        # Generate responses
        responses = []
        for item in tqdm(test_data, desc=f"{model_name.upper()} generation"):
            response = generate_response(model, tokenizer, item['prompt'], config, device)
            responses.append(response)
        aligned_responses[model_name] = responses

        # Score responses
        print(f"Scoring {model_name.upper()} responses...")
        rewards = []
        for prompt, response in tqdm(zip([d['prompt'] for d in test_data], responses),
                                      total=len(test_data), desc=f"Scoring {model_name}"):
            reward = reward_model.score(prompt, response, tokenizer)
            rewards.append(reward)
            model_stats[model_name]['rewards'].append(reward)
            model_stats[model_name]['lengths'].append(len(response.split()))
            model_stats[model_name]['repetitions'].append(compute_repetition_ratio(response))
        aligned_rewards[model_name] = rewards

        # Free memory
        del model
        torch.mps.empty_cache() if device == 'mps' else torch.cuda.empty_cache()

    # Compute pairwise comparisons
    print("\n=== Computing Pairwise Comparisons ===")
    for i in range(len(test_data)):
        # vs Base comparisons
        for model_name in aligned_models:
            key = f'{model_name}_vs_base'
            result = compute_pairwise_comparison(aligned_rewards[model_name][i], base_rewards[i])
            if result == "A":
                pairwise[key][model_name] += 1
            elif result == "B":
                pairwise[key]['base'] += 1
            else:
                pairwise[key]['tie'] += 1

        # DPO vs PPO
        result = compute_pairwise_comparison(aligned_rewards['dpo'][i], aligned_rewards['ppo'][i])
        if result == "A":
            pairwise['dpo_vs_ppo']['dpo'] += 1
        elif result == "B":
            pairwise['dpo_vs_ppo']['ppo'] += 1
        else:
            pairwise['dpo_vs_ppo']['tie'] += 1

        # DPO vs GRPO
        result = compute_pairwise_comparison(aligned_rewards['dpo'][i], aligned_rewards['grpo'][i])
        if result == "A":
            pairwise['dpo_vs_grpo']['dpo'] += 1
        elif result == "B":
            pairwise['dpo_vs_grpo']['grpo'] += 1
        else:
            pairwise['dpo_vs_grpo']['tie'] += 1

        # PPO vs GRPO
        result = compute_pairwise_comparison(aligned_rewards['ppo'][i], aligned_rewards['grpo'][i])
        if result == "A":
            pairwise['ppo_vs_grpo']['ppo'] += 1
        elif result == "B":
            pairwise['ppo_vs_grpo']['grpo'] += 1
        else:
            pairwise['ppo_vs_grpo']['tie'] += 1

        # Store individual result
        all_results.append({
            'prompt': test_data[i]['prompt'][:200] + '...',  # Truncate for readability
            'base_response': base_responses[i][:200] + '...',
            'dpo_response': aligned_responses['dpo'][i][:200] + '...',
            'ppo_response': aligned_responses['ppo'][i][:200] + '...',
            'grpo_response': aligned_responses['grpo'][i][:200] + '...',
            'base_reward': base_rewards[i],
            'dpo_reward': aligned_rewards['dpo'][i],
            'ppo_reward': aligned_rewards['ppo'][i],
            'grpo_reward': aligned_rewards['grpo'][i],
        })

    # Compute summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(test_data),
        'config': config,
        'model_stats': {},
        'pairwise_comparisons': {},
        'win_rates': {},
    }

    for model_name in ['base', 'dpo', 'ppo', 'grpo']:
        rewards = model_stats[model_name]['rewards']
        lengths = model_stats[model_name]['lengths']
        reps = model_stats[model_name]['repetitions']

        summary['model_stats'][model_name] = {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'median_reward': float(np.median(rewards)),
            'mean_length': float(np.mean(lengths)),
            'mean_repetition': float(np.mean(reps)),
        }

    # Compute win rates
    n = len(test_data)
    for key, counts in pairwise.items():
        models = [k for k in counts.keys() if k != 'tie']
        summary['pairwise_comparisons'][key] = counts
        for m in models:
            if m != 'base':
                win_rate = counts[m] / n * 100
                summary['win_rates'][f'{key}_{m}_win_rate'] = win_rate

    # Print results
    print("\n" + "="*60)
    print("EXPERIMENT J RESULTS: DPO vs PPO vs GRPO")
    print("="*60)

    print("\n### Model Statistics ###")
    print(f"{'Model':<10} {'Mean Reward':>12} {'Std':>8} {'Min':>8} {'Max':>8} {'Avg Len':>8}")
    print("-" * 60)
    for model_name in ['base', 'dpo', 'ppo', 'grpo']:
        stats = summary['model_stats'][model_name]
        print(f"{model_name.upper():<10} {stats['mean_reward']:>12.4f} {stats['std_reward']:>8.4f} "
              f"{stats['min_reward']:>8.4f} {stats['max_reward']:>8.4f} {stats['mean_length']:>8.1f}")

    print("\n### Win Rates vs Base ###")
    for model_name in ['dpo', 'ppo', 'grpo']:
        key = f'{model_name}_vs_base'
        wins = pairwise[key][model_name]
        losses = pairwise[key]['base']
        ties = pairwise[key]['tie']
        win_rate = wins / n * 100
        print(f"{model_name.upper()}: {wins} wins, {losses} losses, {ties} ties ({win_rate:.1f}% win rate)")

    print("\n### Head-to-Head Comparisons ###")
    for key in ['dpo_vs_ppo', 'dpo_vs_grpo', 'ppo_vs_grpo']:
        counts = pairwise[key]
        models = [k for k in counts.keys() if k != 'tie']
        print(f"{key}: {models[0].upper()}={counts[models[0]]}, {models[1].upper()}={counts[models[1]]}, Ties={counts['tie']}")

    # Save results
    print("\n### Saving Results ###")

    # Main comparison results
    with open(results_dir / 'comparison_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {results_dir / 'comparison_results.json'}")

    # Pairwise matrix
    with open(results_dir / 'pairwise_matrix.json', 'w') as f:
        json.dump(pairwise, f, indent=2)
    print(f"Saved: {results_dir / 'pairwise_matrix.json'}")

    # Sample responses
    with open(results_dir / 'sample_responses.json', 'w') as f:
        json.dump(all_results[:20], f, indent=2)  # Save first 20 for inspection
    print(f"Saved: {results_dir / 'sample_responses.json'}")

    # Full results
    with open(results_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {results_dir / 'full_results.json'}")

    print("\n" + "="*60)
    print("Experiment J Complete!")
    print("="*60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Experiment J: DPO vs PPO vs GRPO comparison")
    parser.add_argument('--config', type=str, default='configs/experiment_j.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['seed'])

    results = run_comparison(config)

    return results


if __name__ == '__main__':
    main()
