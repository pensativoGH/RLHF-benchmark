#!/usr/bin/env python3
"""
MATH-500 Evaluation Script

Evaluates a model on the MATH-500 benchmark, reporting:
- Overall accuracy
- Per-subject accuracy
- Per-level accuracy
- Format compliance (% with \boxed{})
- Average response length

Usage:
    python scripts/eval_math500.py --model Qwen/Qwen3-0.6B
    python scripts/eval_math500.py --model checkpoints/grpo_rlvr/step_50
    python scripts/eval_math500.py --model Qwen/Qwen3-0.6B --max-problems 50
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.math_loader import load_math500, format_math_prompt
from src.reasoning.math_grader import extract_boxed_answer, grade_answer, reward_rlvr


def get_device():
    """Detect available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: str, device: torch.device):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: torch.device = None,
) -> tuple[str, int]:
    """
    Generate a response for a given prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: The input prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        device: Device to use.

    Returns:
        Tuple of (generated_text, num_tokens).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generation config
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated part
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, len(generated_ids)


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: torch.device = None,
) -> list[tuple[str, int]]:
    """
    Generate responses for a batch of prompts.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompts: List of input prompts.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        device: Device to use.

    Returns:
        List of tuples (generated_text, num_tokens).
    """
    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    # Generation config
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    outputs = model.generate(**inputs, **gen_kwargs)

    # Decode each response
    results = []
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

    for i, (output_ids, prompt_len) in enumerate(zip(outputs, prompt_lens)):
        generated_ids = output_ids[prompt_len:]
        # Remove padding tokens
        generated_ids = generated_ids[generated_ids != tokenizer.pad_token_id]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append((generated_text, len(generated_ids)))

    return results


def evaluate_math500(
    model,
    tokenizer,
    device: torch.device,
    max_problems: int = None,
    skip: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 1,
    verbose: bool = False,
) -> dict:
    """
    Evaluate model on MATH-500 benchmark.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        device: Device to use.
        max_problems: Max problems to evaluate (None = all 500).
        skip: Number of problems to skip (for resuming).
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        batch_size: Batch size for generation (higher = faster but more memory).
        verbose: Print individual results.

    Returns:
        Dictionary with evaluation results.
    """
    # Load dataset
    print("Loading MATH-500 dataset...")
    problems = load_math500()

    # Apply skip and max_problems
    if skip > 0:
        print(f"Skipping first {skip} problems...")
        problems = problems[skip:]

    if max_problems:
        problems = problems[:max_problems]

    print(f"Evaluating on {len(problems)} problems (batch_size={batch_size})...")

    # Tracking
    results = []
    correct_by_subject = defaultdict(int)
    total_by_subject = defaultdict(int)
    correct_by_level = defaultdict(int)
    total_by_level = defaultdict(int)
    format_compliant = 0
    total_tokens = 0

    # Process in batches
    num_batches = (len(problems) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(problems))
        batch_problems = problems[batch_start:batch_end]

        # Format prompts
        prompts = [format_math_prompt(p.problem) for p in batch_problems]

        # Generate responses
        if batch_size == 1:
            # Single item - use original function
            response, num_tokens = generate_response(
                model, tokenizer, prompts[0], max_new_tokens, temperature, device
            )
            batch_responses = [(response, num_tokens)]
        else:
            # Batch generation
            batch_responses = generate_batch(
                model, tokenizer, prompts, max_new_tokens, temperature, device
            )

        # Process each response
        for problem, (response, num_tokens) in zip(batch_problems, batch_responses):
            total_tokens += num_tokens

            # Extract and grade
            extracted = extract_boxed_answer(response)
            has_boxed = extracted is not None
            is_correct = grade_answer(extracted, problem.answer) if has_boxed else False

            # Track results
            result = {
                "unique_id": problem.unique_id,
                "subject": problem.subject,
                "level": problem.level,
                "problem": problem.problem[:200],
                "ground_truth": problem.answer,
                "extracted_answer": extracted,
                "has_boxed": has_boxed,
                "is_correct": is_correct,
                "response_length": num_tokens,
            }
            results.append(result)

            if has_boxed:
                format_compliant += 1

            total_by_subject[problem.subject] += 1
            total_by_level[problem.level] += 1

            if is_correct:
                correct_by_subject[problem.subject] += 1
                correct_by_level[problem.level] += 1

            if verbose:
                status = "✓" if is_correct else "✗"
                boxed = "📦" if has_boxed else "⚠️"
                print(f"{status} {boxed} [{problem.subject}][{problem.level}] "
                      f"GT: {problem.answer} | Pred: {extracted}")

    # Compute summary stats
    total = len(problems)
    correct = sum(1 for r in results if r["is_correct"])

    summary = {
        "model": str(model.config._name_or_path),
        "timestamp": datetime.now().isoformat(),
        "total_problems": total,
        "correct": correct,
        "accuracy": correct / total,
        "format_compliant": format_compliant,
        "format_compliance_rate": format_compliant / total,
        "avg_response_length": total_tokens / total,
        "by_subject": {
            subject: {
                "correct": correct_by_subject[subject],
                "total": total_by_subject[subject],
                "accuracy": correct_by_subject[subject] / total_by_subject[subject]
            }
            for subject in sorted(total_by_subject.keys())
        },
        "by_level": {
            level: {
                "correct": correct_by_level[level],
                "total": total_by_level[level],
                "accuracy": correct_by_level[level] / total_by_level[level]
            }
            for level in sorted(total_by_level.keys(), key=lambda x: int(x) if x.isdigit() else x)
        },
        "results": results,
    }

    return summary


def print_results(summary: dict):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("MATH-500 Evaluation Results")
    print("=" * 60)

    print(f"\nModel: {summary['model']}")
    print(f"Total: {summary['total_problems']} problems")

    acc_pct = summary['accuracy'] * 100
    print(f"\nOverall Accuracy: {summary['correct']}/{summary['total_problems']} = {acc_pct:.1f}%")

    print("\nBy Subject:")
    for subject, stats in summary['by_subject'].items():
        acc = stats['accuracy'] * 100
        print(f"  {subject:20s} {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}%")

    print("\nBy Difficulty Level:")
    for level, stats in summary['by_level'].items():
        acc = stats['accuracy'] * 100
        level_str = f"Level {level}"
        print(f"  {level_str:20s} {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}%")

    fmt_pct = summary['format_compliance_rate'] * 100
    print(f"\nFormat Compliance: {summary['format_compliant']}/{summary['total_problems']} = {fmt_pct:.1f}% (responses with \\boxed{{}})")
    print(f"Avg Response Length: {summary['avg_response_length']:.0f} tokens")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MATH-500")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum problems to evaluate (default: all 500)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N problems (for resuming interrupted runs)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (higher = faster but more memory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print individual problem results",
    )

    args = parser.parse_args()

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model(args.model, device)

    # Evaluate
    summary = evaluate_math500(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_problems=args.max_problems,
        skip=args.skip,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # Print results
    print_results(summary)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {output_path}")
    else:
        # Default output path
        model_name = args.model.replace("/", "_")
        output_path = Path("results") / f"math500_eval_{model_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
