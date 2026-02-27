#!/usr/bin/env python3
"""
Evaluate rasbt's Qwen3-0.6B-base model on MATH-500.

This script uses rasbt's model implementation and grading code
to verify we get the expected ~15% baseline accuracy.

Usage:
    python rasbt_baseline/eval_rasbt_model.py
    python rasbt_baseline/eval_rasbt_model.py --max-problems 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add src to path for rasbt's code
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import rasbt's modules (after fixing path)
from qwen3 import Qwen3Model, Qwen3Tokenizer, QWEN_CONFIG_06_B, KVCache


def get_device():
    """Detect available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rasbt_model(weights_dir: str, device: torch.device):
    """Load rasbt's Qwen3 model and tokenizer."""
    weights_path = Path(weights_dir)

    model_path = weights_path / "qwen3-0.6B-base.pth"
    tokenizer_path = weights_path / "tokenizer-base.json"

    print(f"Loading model from {model_path}...")

    # Initialize model
    model = Qwen3Model(QWEN_CONFIG_06_B)

    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=str(tokenizer_path))

    return model, tokenizer


def render_prompt(problem: str) -> str:
    """Format prompt using rasbt's exact template."""
    return (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{problem}\n\nAnswer:"
    )


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 512,
) -> str:
    """Generate text using rasbt's model with KV cache."""
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)

    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    generated_ids = []

    # Initial forward pass (prefill)
    out = model(input_ids, cache=cache)[:, -1]

    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token.item())

        # Generate next token
        out = model(next_token, cache=cache)[:, -1]

    return tokenizer.decode(generated_ids)


def get_last_boxed(text: str) -> str:
    """Extract content from the last \\boxed{...} in text."""
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    current_idx = boxed_start_idx + len(r"\boxed")

    # Skip whitespace
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    # Parse braces with nesting
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:
        return None

    return text[content_start_idx:current_idx-1]


def normalize_answer(text: str) -> str:
    """Simple normalization for answer comparison."""
    if text is None:
        return ""

    result = text.strip().lower()

    # Remove common LaTeX formatting
    result = result.replace("\\left", "").replace("\\right", "")
    result = result.replace("\\,", " ").replace("\\;", " ")
    result = result.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    result = result.replace("{", "").replace("}", "")
    result = " ".join(result.split())
    result = result.rstrip(".,;")

    return result


def grade_answer_simple(predicted: str, ground_truth: str) -> bool:
    """Simple answer grading - exact match after normalization."""
    if predicted is None or ground_truth is None:
        return False

    pred_norm = normalize_answer(predicted)
    truth_norm = normalize_answer(ground_truth)

    # Exact match
    if pred_norm == truth_norm:
        return True

    # Try numeric comparison
    try:
        pred_num = float(pred_norm)
        truth_num = float(truth_norm)
        if abs(pred_num - truth_num) < 1e-6:
            return True
    except ValueError:
        pass

    return False


def load_math500():
    """Load MATH-500 from HuggingFace."""
    from datasets import load_dataset

    print("Loading MATH-500 dataset from HuggingFace...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    problems = []
    for item in dataset:
        problems.append({
            "problem": item["problem"],
            "answer": item["answer"],
            "subject": item["subject"],
            "level": str(item["level"]),
            "unique_id": item["unique_id"],
        })

    return problems


def evaluate(
    model,
    tokenizer,
    device: torch.device,
    problems: list,
    max_new_tokens: int = 512,
    verbose: bool = False,
) -> dict:
    """Evaluate model on MATH-500."""
    num_correct = 0
    num_format_ok = 0
    total_tokens = 0

    results = []
    start_time = time.time()

    for i, problem in enumerate(problems):
        prompt = render_prompt(problem["problem"])

        # Generate response
        response = generate_text(
            model, tokenizer, prompt, device, max_new_tokens
        )

        # Count tokens
        num_tokens = len(tokenizer.encode(response))
        total_tokens += num_tokens

        # Extract and grade
        extracted = get_last_boxed(response)
        has_boxed = extracted is not None
        is_correct = grade_answer_simple(extracted, problem["answer"]) if has_boxed else False

        if has_boxed:
            num_format_ok += 1
        if is_correct:
            num_correct += 1

        results.append({
            "unique_id": problem["unique_id"],
            "subject": problem["subject"],
            "level": problem["level"],
            "ground_truth": problem["answer"],
            "extracted": extracted,
            "is_correct": is_correct,
            "has_boxed": has_boxed,
        })

        # Progress
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(problems) - i - 1) if i > 0 else 0
        print(
            f"\rProgress: {i+1}/{len(problems)} | "
            f"Correct: {num_correct} ({100*num_correct/(i+1):.1f}%) | "
            f"ETA: {eta/60:.1f}min",
            end="", flush=True
        )

        if verbose and (i + 1) % 10 == 0:
            print(f"\n  Last answer: GT={problem['answer']}, Pred={extracted}, Correct={is_correct}")

    print()  # Newline after progress

    total = len(problems)
    return {
        "total": total,
        "correct": num_correct,
        "accuracy": num_correct / total,
        "format_compliant": num_format_ok,
        "format_compliance_rate": num_format_ok / total,
        "avg_tokens": total_tokens / total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate rasbt's Qwen3 model on MATH-500")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="rasbt_baseline/weights",
        help="Directory containing model weights",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Max problems to evaluate (default: all 500)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per problem",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rasbt_baseline/results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model, tokenizer = load_rasbt_model(args.weights_dir, device)
    print(f"Model loaded successfully")

    # Load data
    problems = load_math500()
    if args.max_problems:
        problems = problems[:args.max_problems]
    print(f"Evaluating on {len(problems)} problems...")

    # Evaluate
    summary = evaluate(
        model, tokenizer, device, problems,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )

    # Print results
    print("\n" + "=" * 60)
    print("MATH-500 Evaluation Results (rasbt's Qwen3-0.6B-base)")
    print("=" * 60)
    print(f"Total: {summary['total']} problems")
    print(f"Accuracy: {summary['correct']}/{summary['total']} = {100*summary['accuracy']:.1f}%")
    print(f"Format Compliance: {summary['format_compliant']}/{summary['total']} = {100*summary['format_compliance_rate']:.1f}%")
    print(f"Avg Response Length: {summary['avg_tokens']:.0f} tokens")
    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
