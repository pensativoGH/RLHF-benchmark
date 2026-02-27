#!/usr/bin/env python3
"""
GRPO Training using rasbt's exact implementation.

This is a standalone script that runs rasbt's GRPO training code
with the local qwen3-0.6B-base model weights.
"""

import argparse
import time
from pathlib import Path
import sys

import torch

# Add local src path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qwen3 import Qwen3Model, Qwen3Tokenizer, QWEN_CONFIG_06_B, KVCache

SCRIPT_NAME = Path(__file__).stem
LOG_PATH = Path(__file__).parent / "logs" / f"{SCRIPT_NAME}_outputs.txt"
METRICS_LOG_PATH = Path(__file__).parent / "logs" / f"{SCRIPT_NAME}_metrics.txt"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / SCRIPT_NAME


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


def top_p_filter(probas, top_p):
    """Apply top-p (nucleus) sampling filter."""
    if top_p is None or top_p >= 1.0:
        return probas

    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_probas, dim=1)
    mask = cumsum - sorted_probas > top_p
    sorted_probas[mask] = 0.0
    sorted_probas /= sorted_probas.sum(dim=1, keepdim=True)
    probas = torch.zeros_like(probas).scatter_(1, sorted_idx, sorted_probas)
    return probas


def load_math_train():
    """Load MATH training data from cache."""
    import json

    # Check multiple possible cache locations
    cache_paths = [
        Path(__file__).parent / "data" / "math_train.json",
        Path(__file__).parent.parent / "data" / "math_train.json",
    ]

    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"Loading training data from {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)

    # If not cached, download
    import requests
    cache_path = cache_paths[0]
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    url = "https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()

    with open(cache_path, "w") as f:
        json.dump(data, f)

    return data


def load_math500_test():
    """Load MATH-500 test data."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return list(ds)


def render_prompt(prompt):
    """Format a math prompt using rasbt's template."""
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template


def extract_final_candidate(text, fallback=None):
    """Extract the final \\boxed{...} answer from text."""
    import re

    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return fallback

    last_match = matches[-1]
    start_idx = last_match.end()

    depth = 1
    idx = start_idx
    while idx < len(text) and depth > 0:
        if text[idx] == '{':
            depth += 1
        elif text[idx] == '}':
            depth -= 1
        idx += 1

    if depth != 0:
        return fallback

    return text[start_idx:idx - 1].strip()


def grade_answer(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    if predicted is None or ground_truth is None:
        return False

    pred = predicted.strip().lower()
    truth = ground_truth.strip().lower()

    if pred == truth:
        return True

    # Try numeric comparison
    try:
        pred_num = float(pred.replace(",", ""))
        truth_num = float(truth.replace(",", ""))
        return abs(pred_num - truth_num) < 1e-6
    except ValueError:
        pass

    return False


def load_model_and_tokenizer(weights_dir, device):
    """Load model and tokenizer."""
    weights_path = Path(weights_dir)
    model_path = weights_path / "qwen3-0.6B-base.pth"
    tokenizer_path = weights_path / "tokenizer-base.json"

    print(f"Loading model from {model_path}...")
    model = Qwen3Model(QWEN_CONFIG_06_B)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=str(tokenizer_path))

    return model, tokenizer


@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    """Sample a response from the model."""
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
    )

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

    generated = []
    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits = logits / temperature

        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(probas.cpu(), num_samples=1).to(device)

        if (
            tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id
        ):
            break
        generated.append(next_token.item())
        logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype)]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


def sequence_logprob(model, token_ids, prompt_len):
    """Compute summed log probability of response tokens."""
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    targets = token_ids[1:]
    selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)
    return selected[prompt_len - 1:].sum()


def reward_rlvr(answer_text, ground_truth):
    """Compute RLVR reward."""
    extracted = extract_final_candidate(answer_text, fallback=None)
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)


def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=4,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    """Compute GRPO loss for one example."""
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        logp = sequence_logprob(model, token_ids, prompt_len)
        reward = reward_rlvr(text, example["answer"])

        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append({
            "text": text,
            "reward": reward,
            "gen_len": token_ids.numel() - prompt_len,
        })

    if was_training:
        model.train()

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    logps = torch.stack(roll_logps)

    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }


def evaluate_math500(model, tokenizer, device, data, max_new_tokens=512):
    """Evaluate model on MATH-500 subset."""
    model.eval()
    correct = 0

    for item in data:
        prompt = render_prompt(item["problem"])
        _, _, response = sample_response(
            model, tokenizer, prompt, device,
            max_new_tokens=max_new_tokens,
            temperature=0,  # Greedy for eval
            top_p=1.0,
        )
        extracted = extract_final_candidate(response, fallback=None)
        if extracted and grade_answer(extracted, item["answer"]):
            correct += 1

    return correct, len(data), correct / len(data) if data else 0


def append_step_metrics(step_idx, total_steps, loss, reward_avg, tokens_per_sec):
    """Log metrics to file."""
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[Step {step_idx}/{total_steps}] "
            f"loss={loss:.4f} reward_avg={reward_avg:.3f} "
            f"tokens_per_sec={tokens_per_sec:.1f}\n"
        )


def append_sample_logs(step_idx, samples, max_samples=3):
    """Log sample outputs to file."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[Step {step_idx}] sample outputs\n")
        for i, sample in enumerate(samples[:max_samples]):
            text = sample["text"].replace("\n", "\\n")
            f.write(
                f"  {i+1}) reward={sample['reward']:.3f} "
                f"len={sample['gen_len']}: {text[:200]}\n"
            )
        f.write("\n")


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = checkpoint_dir / f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    math500_eval_data,
    device,
    steps=None,
    num_rollouts=8,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=CHECKPOINT_DIR,
    eval_on_checkpoint=False,
    eval_max_items=20,
):
    """Train with GRPO using RLVR rewards."""
    if steps is None:
        steps = len(math_data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    current_step = 0

    try:
        for step in range(steps):
            step_start = time.perf_counter()
            current_step = step + 1
            example = math_data[step % len(math_data)]

            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            optimizer.zero_grad()
            stats["loss_tensor"].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_time = time.perf_counter() - step_start
            step_tokens = sum(sample["gen_len"] for sample in stats["samples"])
            num_correct = sum(stats["rewards"])

            # Clear MPS memory aggressively
            del stats["loss_tensor"]
            if device.type == "mps":
                torch.mps.empty_cache()
                import gc
                gc.collect()
            tokens_per_sec = step_tokens / step_time if step_time > 0 else 0.0

            append_step_metrics(current_step, steps, stats["loss"], reward_avg, tokens_per_sec)

            if current_step % 10 == 0:
                append_sample_logs(current_step, stats["samples"])

            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")

                if eval_on_checkpoint and math500_eval_data:
                    was_training = model.training
                    model.eval()
                    subset = math500_eval_data[:eval_max_items] if eval_max_items else math500_eval_data
                    correct, total, acc = evaluate_math500(
                        model, tokenizer, device, subset, max_new_tokens
                    )
                    print(f"MATH-500 eval @ step {current_step}: acc={acc:.3f} ({correct}/{total})")
                    if was_training:
                        model.train()

            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"correct={int(num_correct)}/{num_rollouts} "
                f"tokens_per_sec={tokens_per_sec:.1f}"
            )

    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RLVR GRPO on MATH dataset.")
    parser.add_argument("--weights-dir", type=str, default="weights",
                        help="Directory containing model weights")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of training steps")
    parser.add_argument("--num-rollouts", type=int, default=8,
                        help="Number of rollouts per step")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum tokens to generate per rollout")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling cutoff")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval-on-checkpoint", action="store_true",
                        help="Run MATH-500 eval when saving checkpoints")
    parser.add_argument("--eval-max-items", type=int, default=50,
                        help="Max items for checkpoint evaluation")

    args = parser.parse_args()

    device = get_device()

    print("\nLoading training data...")
    math_data = load_math_train()
    print(f"Loaded {len(math_data)} training problems")

    print("\nLoading MATH-500 for evaluation...")
    math500_data = load_math500_test()
    print(f"Loaded {len(math500_data)} eval problems")

    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.weights_dir, device)
    print("Model loaded successfully")

    # Initial evaluation
    print("\nRunning initial evaluation on 50 problems...")
    model.eval()
    correct, total, acc = evaluate_math500(
        model, tokenizer, device, math500_data[:50], args.max_new_tokens
    )
    print(f"Initial accuracy: {acc*100:.1f}% ({correct}/{total})")
    model.train()

    print(f"\n{'='*60}")
    print("Starting GRPO + RLVR Training (rasbt's code)")
    print(f"{'='*60}")
    print(f"Steps: {args.steps}")
    print(f"Rollouts per prompt: {args.num_rollouts}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"{'='*60}\n")

    trained = train_rlvr_grpo(
        model=model,
        tokenizer=tokenizer,
        math_data=math_data,
        math500_eval_data=math500_data,
        device=device,
        steps=args.steps,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        checkpoint_every=args.checkpoint_every,
        eval_on_checkpoint=args.eval_on_checkpoint,
        eval_max_items=args.eval_max_items,
    )

    # Final evaluation
    print("\nRunning final evaluation on 50 problems...")
    model.eval()
    correct, total, acc = evaluate_math500(
        model, tokenizer, device, math500_data[:50], args.max_new_tokens
    )
    print(f"Final accuracy: {acc*100:.1f}% ({correct}/{total})")

    # Save final model
    final_path = CHECKPOINT_DIR / "qwen3-0.6B-rlvr-grpo-final.pth"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(trained.state_dict(), final_path)
    print(f"\nSaved final model to {final_path}")
