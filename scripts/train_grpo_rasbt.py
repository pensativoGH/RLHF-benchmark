#!/usr/bin/env python3
"""
GRPO + RLVR Training Script using rasbt's Qwen3-0.6B-base model

Implements Group Relative Policy Optimization with Verifiable Rewards
for training on MATH dataset using rasbt's custom Qwen3 implementation.

Key features:
- Uses rasbt's Qwen3-0.6B-base model (16% baseline accuracy)
- Binary RLVR rewards (correct = 1, wrong = 0)
- Summed sequence-level log probabilities
- Simple policy gradient (no clipping, no KL by default)
- Full fine-tuning

Usage:
    python scripts/train_grpo_rasbt.py --num-steps 50 --num-rollouts 4
    python scripts/train_grpo_rasbt.py --config configs/grpo_rlvr.yaml
"""

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "rasbt_baseline" / "src"))

# Import rasbt's model
from qwen3 import Qwen3Model, Qwen3Tokenizer, QWEN_CONFIG_06_B, KVCache

# Import our data loaders and graders
from src.data.math_loader import (
    load_math_train,
    load_math500,
    format_math_prompt,
    MathTrainIterator,
)
from src.reasoning.math_grader import (
    reward_rlvr,
    compute_advantages,
    extract_boxed_answer,
    grade_answer,
)


def get_device():
    """Detect available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_rasbt_model(weights_dir: str, device: torch.device):
    """
    Load rasbt's Qwen3-0.6B-base model.

    Args:
        weights_dir: Directory containing model weights and tokenizer.
        device: Device to load model on.

    Returns:
        Tuple of (model, tokenizer).
    """
    weights_path = Path(weights_dir)
    model_path = weights_path / "qwen3-0.6B-base.pth"
    tokenizer_path = weights_path / "tokenizer-base.json"

    print(f"Loading rasbt's Qwen3-0.6B-base model...")
    print(f"  Model weights: {model_path}")
    print(f"  Tokenizer: {tokenizer_path}")

    # Initialize model
    model = Qwen3Model(QWEN_CONFIG_06_B)

    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    # Load tokenizer
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=str(tokenizer_path))

    return model, tokenizer


def sequence_logprob(
    model,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Compute SUMMED log probability for a sequence (no gradients).

    Args:
        model: The Qwen3Model.
        input_ids: Full sequence (prompt + response), shape [seq_len].
        prompt_len: Length of the prompt portion.

    Returns:
        Scalar tensor with summed log probability of response tokens.
    """
    with torch.no_grad():
        # rasbt's model returns logits directly (not in .logits attribute)
        logits = model(input_ids.unsqueeze(0)).squeeze(0).float()

    # Compute log probabilities
    logprobs = torch.log_softmax(logits, dim=-1)

    # Get log probs of actual tokens (shifted by 1 for autoregressive)
    selected_logprobs = logprobs[:-1].gather(
        1, input_ids[1:].unsqueeze(-1)
    ).squeeze(-1)

    # Sum only the response tokens (after prompt)
    response_logprobs = selected_logprobs[prompt_len - 1:]
    return torch.sum(response_logprobs)


def sequence_logprob_with_grad(
    model,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Compute SUMMED log probability with gradients enabled.

    Args:
        model: The Qwen3Model.
        input_ids: Full sequence (prompt + response), shape [seq_len].
        prompt_len: Length of the prompt portion.

    Returns:
        Scalar tensor with summed log probability of response tokens.
    """
    # rasbt's model returns logits directly
    logits = model(input_ids.unsqueeze(0)).squeeze(0).float()

    logprobs = torch.log_softmax(logits, dim=-1)

    selected_logprobs = logprobs[:-1].gather(
        1, input_ids[1:].unsqueeze(-1)
    ).squeeze(-1)

    response_logprobs = selected_logprobs[prompt_len - 1:]
    return torch.sum(response_logprobs)


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    use_sampling: bool = True,
) -> tuple[torch.Tensor, str, int]:
    """
    Generate a response using rasbt's model with KV cache.

    Args:
        model: The Qwen3Model.
        tokenizer: The Qwen3Tokenizer.
        prompt: Input prompt text.
        device: Device to use.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        use_sampling: Whether to use sampling (True) or greedy (False).

    Returns:
        Tuple of (full_ids, response_text, prompt_len).
    """
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor(prompt_ids, device=device).unsqueeze(0)

    # Setup KV cache
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    generated_ids = list(prompt_ids)

    # Initial forward pass (prefill)
    logits = model(input_ids, cache=cache)[:, -1]

    for _ in range(max_new_tokens):
        if use_sampling and temperature > 0:
            # Apply temperature
            logits = logits / temperature

            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        next_token_id = next_token.item()

        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id)

        # Generate next logits
        logits = model(next_token, cache=cache)[:, -1]

    # Create full tensor
    full_ids = torch.tensor(generated_ids, device=device)

    # Decode response (only the generated part)
    response_ids = generated_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids)

    return full_ids, response_text, prompt_len


def generate_rollouts(
    model,
    tokenizer,
    prompt: str,
    num_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[dict]:
    """
    Generate multiple rollouts for a given prompt.

    Args:
        model: The Qwen3Model.
        tokenizer: The Qwen3Tokenizer.
        prompt: The input prompt.
        num_rollouts: Number of responses to generate.
        max_new_tokens: Maximum tokens per response.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        device: Device to use.

    Returns:
        List of rollout dictionaries.
    """
    rollouts = []

    for _ in range(num_rollouts):
        full_ids, response_text, prompt_len = generate_response(
            model, tokenizer, prompt, device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_sampling=True,
        )

        rollouts.append({
            "input_ids": full_ids.clone(),  # Clone to avoid inference_mode restriction
            "prompt_len": prompt_len,
            "response_text": response_text,
            "response_len": len(full_ids) - prompt_len,
        })

    return rollouts


def grpo_step(
    model,
    optimizer,
    rollouts: list[dict],
    rewards: list[float],
    gradient_clip: float = 1.0,
    ref_model=None,
    kl_coeff: float = 0.0,
) -> dict:
    """
    Perform one GRPO update step.

    Loss = -(advantages.detach() * logps).mean() + kl_coeff * kl_penalty

    Args:
        model: The Qwen3Model (being trained).
        optimizer: The optimizer.
        rollouts: List of rollout dictionaries.
        rewards: List of rewards for each rollout.
        gradient_clip: Max gradient norm.
        ref_model: Reference model for KL penalty (optional).
        kl_coeff: KL penalty coefficient (0 = no penalty).

    Returns:
        Dictionary with training metrics.
    """
    # Compute advantages
    advantages = compute_advantages(rewards)

    # Note: We do NOT skip when all advantages are zero.
    # Zero advantages will produce zero loss/gradients naturally.

    # Compute log probabilities with gradients
    logps = []
    response_lengths = []
    for rollout in rollouts:
        logp = sequence_logprob_with_grad(
            model,
            rollout["input_ids"],
            rollout["prompt_len"],
        )
        logps.append(logp)
        response_lengths.append(rollout["response_len"])

    logps = torch.stack(logps)
    advantages_tensor = torch.tensor(advantages, device=logps.device, dtype=logps.dtype)

    # Policy gradient loss: -(A * logp).mean()
    pg_loss = -(advantages_tensor.detach() * logps).mean()

    # KL penalty (if enabled)
    kl_loss = torch.tensor(0.0, device=logps.device)
    mean_kl = 0.0
    if ref_model is not None and kl_coeff > 0:
        with torch.no_grad():
            ref_logps = []
            for rollout in rollouts:
                ref_logp = sequence_logprob(
                    ref_model, rollout["input_ids"], rollout["prompt_len"]
                )
                ref_logps.append(ref_logp)
            ref_logps = torch.stack(ref_logps)

        # Per-token normalized KL
        lengths = torch.tensor(response_lengths, device=logps.device, dtype=logps.dtype)
        lengths = torch.clamp(lengths, min=1)
        per_token_kl = (logps - ref_logps) / lengths
        kl_loss = kl_coeff * per_token_kl.mean()
        mean_kl = per_token_kl.mean().item()

    # Total loss
    loss = pg_loss + kl_loss

    # Backward and update
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

    optimizer.step()

    result = {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "mean_reward": sum(rewards) / len(rewards),
        "mean_advantage": sum(abs(a) for a in advantages) / len(advantages),
        "mean_logp": logps.mean().item(),
        "skipped": False,
    }

    if kl_coeff > 0:
        result["kl_loss"] = kl_loss.item()
        result["mean_kl"] = mean_kl

    return result


def evaluate_subset(
    model,
    tokenizer,
    problems,
    max_new_tokens: int,
    device: torch.device,
) -> dict:
    """Quick evaluation on a subset of MATH-500."""
    correct = 0
    format_ok = 0

    for problem in problems:
        prompt = format_math_prompt(problem.problem)

        _, response, _ = generate_response(
            model, tokenizer, prompt, device,
            max_new_tokens=max_new_tokens,
            use_sampling=False,  # Greedy for eval
        )

        extracted = extract_boxed_answer(response)
        if extracted is not None:
            format_ok += 1
            if grade_answer(extracted, problem.answer):
                correct += 1

    return {
        "accuracy": correct / len(problems),
        "format_compliance": format_ok / len(problems),
        "correct": correct,
        "total": len(problems),
    }


def train(
    weights_dir: str = "rasbt_baseline/weights",
    num_steps: int = 50,
    num_rollouts: int = 4,
    max_new_tokens: int = 256,  # Reduced from 512 for MPS memory
    temperature: float = 0.8,
    top_p: float = 0.9,
    learning_rate: float = 1e-5,
    gradient_clip: float = 1.0,
    use_kl_penalty: bool = False,
    kl_coeff: float = 0.02,
    save_every: int = 5,
    eval_every: int = 10,
    eval_problems: int = 50,
    output_dir: str = "checkpoints/grpo_rasbt",
    train_cache: str = "data/math_train.json",
    seed: int = 42,
):
    """
    Train rasbt's model with GRPO + RLVR.

    Args:
        weights_dir: Directory containing rasbt's model weights.
        num_steps: Number of training steps.
        num_rollouts: Rollouts per prompt.
        max_new_tokens: Max tokens per response.
        temperature: Sampling temperature.
        top_p: Top-p for sampling.
        learning_rate: Learning rate.
        gradient_clip: Max gradient norm.
        use_kl_penalty: Whether to use KL penalty.
        kl_coeff: KL penalty coefficient.
        save_every: Save checkpoint every N steps.
        eval_every: Evaluate every N steps.
        eval_problems: Number of problems for quick eval.
        output_dir: Directory for checkpoints.
        train_cache: Path to cache training data.
        seed: Random seed.
    """
    # Setup
    torch.manual_seed(seed)
    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_rasbt_model(weights_dir, device)
    model.train()

    # Reference model for KL (if enabled)
    ref_model = None
    if use_kl_penalty:
        print("Creating reference model for KL penalty...")
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    # Load data
    print("\nLoading training data...")
    Path(train_cache).parent.mkdir(parents=True, exist_ok=True)
    train_problems = load_math_train(cache_path=train_cache)
    print(f"Loaded {len(train_problems)} training problems")

    train_iter = MathTrainIterator(train_problems, shuffle=True, seed=seed)

    # Load eval data
    print("\nLoading MATH-500 for evaluation...")
    eval_problems_list = load_math500()[:eval_problems]
    print(f"Using {len(eval_problems_list)} problems for quick eval")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training metrics
    metrics_log = []

    print(f"\n{'='*60}")
    print("Starting GRPO + RLVR Training (rasbt's base model)")
    print(f"{'='*60}")
    print(f"Model: rasbt's Qwen3-0.6B-base")
    print(f"Steps: {num_steps}")
    print(f"Rollouts per prompt: {num_rollouts}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Learning rate: {learning_rate}")
    print(f"KL penalty: {'Yes' if use_kl_penalty else 'No'}")
    print(f"{'='*60}\n")

    # Initial evaluation
    print("Running initial evaluation...")
    model.eval()
    initial_eval = evaluate_subset(
        model, tokenizer, eval_problems_list, max_new_tokens, device
    )
    model.train()
    print(f"Initial accuracy: {initial_eval['accuracy']*100:.1f}% "
          f"({initial_eval['correct']}/{initial_eval['total']})")
    print(f"Initial format compliance: {initial_eval['format_compliance']*100:.1f}%\n")

    # Training loop
    for step in tqdm(range(1, num_steps + 1), desc="Training"):
        # Get next problem
        problem = next(train_iter)
        prompt = format_math_prompt(problem.problem)

        # Generate rollouts (in eval mode for generation)
        model.eval()
        rollouts = generate_rollouts(
            model, tokenizer, prompt, num_rollouts,
            max_new_tokens, temperature, top_p, device
        )
        model.train()

        # Compute rewards
        rewards = [
            reward_rlvr(r["response_text"], problem.answer)
            for r in rollouts
        ]

        # GRPO update
        step_metrics = grpo_step(
            model, optimizer, rollouts, rewards, gradient_clip,
            ref_model=ref_model if use_kl_penalty else None,
            kl_coeff=kl_coeff if use_kl_penalty else 0.0,
        )

        # Clear memory (important for MPS)
        del rollouts
        if device.type == "mps":
            torch.mps.empty_cache()

        # Log metrics
        step_metrics["step"] = step
        step_metrics["num_correct"] = sum(1 for r in rewards if r > 0)
        step_metrics["subject"] = problem.subject
        step_metrics["level"] = problem.level
        metrics_log.append(step_metrics)

        # Print progress
        if step % 5 == 0 or step == 1:
            msg = (f"Step {step}: loss={step_metrics['loss']:.4f}, "
                   f"reward={step_metrics['mean_reward']:.2f}, "
                   f"correct={step_metrics['num_correct']}/{num_rollouts}")
            if use_kl_penalty and "mean_kl" in step_metrics:
                msg += f", kl={step_metrics['mean_kl']:.4f}"
            tqdm.write(msg)

        # Evaluation
        if step % eval_every == 0:
            model.eval()
            eval_result = evaluate_subset(
                model, tokenizer, eval_problems_list, max_new_tokens, device
            )
            model.train()
            tqdm.write(
                f"\n  [Eval] Accuracy: {eval_result['accuracy']*100:.1f}% "
                f"({eval_result['correct']}/{eval_result['total']}), "
                f"Format: {eval_result['format_compliance']*100:.1f}%\n"
            )
            metrics_log[-1]["eval_accuracy"] = eval_result["accuracy"]
            metrics_log[-1]["eval_format"] = eval_result["format_compliance"]

        # Save checkpoint
        if step % save_every == 0:
            ckpt_path = output_path / f"step_{step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path / "model.pth")
            tqdm.write(f"  Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = output_path / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path / "model.pth")
    print(f"\nSaved final model to {final_path}")

    # Save metrics
    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Final evaluation
    print("\nRunning final evaluation...")
    model.eval()
    final_eval = evaluate_subset(
        model, tokenizer, eval_problems_list, max_new_tokens, device
    )
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Initial accuracy: {initial_eval['accuracy']*100:.1f}%")
    print(f"Final accuracy: {final_eval['accuracy']*100:.1f}%")
    print(f"Improvement: +{(final_eval['accuracy'] - initial_eval['accuracy'])*100:.1f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="GRPO + RLVR Training with rasbt's model")

    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--weights-dir", type=str, default="rasbt_baseline/weights")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--use-kl-penalty", action="store_true")
    parser.add_argument("--kl-coeff", type=float, default=0.02)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-problems", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="checkpoints/grpo_rasbt")
    parser.add_argument("--train-cache", type=str, default="data/math_train.json")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Use config values as defaults, CLI args override
        weights_dir = config.get("model", {}).get("weights_dir", args.weights_dir)
        num_steps = config.get("training", {}).get("num_steps", args.num_steps)
        num_rollouts = config.get("generation", {}).get("num_rollouts", args.num_rollouts)
        max_new_tokens = config.get("generation", {}).get("max_new_tokens", args.max_new_tokens)
        temperature = config.get("generation", {}).get("temperature", args.temperature)
        top_p = config.get("generation", {}).get("top_p", args.top_p)
        learning_rate = config.get("training", {}).get("learning_rate", args.learning_rate)
        gradient_clip = config.get("training", {}).get("gradient_clip", args.gradient_clip)
        use_kl_penalty = config.get("training", {}).get("use_kl_penalty", args.use_kl_penalty)
        kl_coeff = config.get("training", {}).get("kl_coeff", args.kl_coeff)
        save_every = config.get("checkpoint", {}).get("save_every", args.save_every)
        eval_every = config.get("eval", {}).get("eval_every", args.eval_every)
        eval_problems = config.get("eval", {}).get("eval_problems", args.eval_problems)
        output_dir = config.get("checkpoint", {}).get("output_dir", args.output_dir)
        train_cache = config.get("data", {}).get("train_cache", args.train_cache)
        seed = config.get("data", {}).get("seed", args.seed)
    else:
        weights_dir = args.weights_dir
        num_steps = args.num_steps
        num_rollouts = args.num_rollouts
        max_new_tokens = args.max_new_tokens
        temperature = args.temperature
        top_p = args.top_p
        learning_rate = args.learning_rate
        gradient_clip = args.gradient_clip
        use_kl_penalty = args.use_kl_penalty
        kl_coeff = args.kl_coeff
        save_every = args.save_every
        eval_every = args.eval_every
        eval_problems = args.eval_problems
        output_dir = args.output_dir
        train_cache = args.train_cache
        seed = args.seed

    train(
        weights_dir=weights_dir,
        num_steps=num_steps,
        num_rollouts=num_rollouts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        learning_rate=learning_rate,
        gradient_clip=gradient_clip,
        use_kl_penalty=use_kl_penalty,
        kl_coeff=kl_coeff,
        save_every=save_every,
        eval_every=eval_every,
        eval_problems=eval_problems,
        output_dir=output_dir,
        train_cache=train_cache,
        seed=seed,
    )


if __name__ == "__main__":
    main()
