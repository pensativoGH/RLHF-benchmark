#!/usr/bin/env python3
"""Visualization script for reward model training results."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_training_logs(results_dir: Path) -> list:
    """Load training logs from JSON Lines file."""
    log_path = results_dir / "logs" / "train_metrics.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Training logs not found: {log_path}")

    logs = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs


def load_final_metrics(results_dir: Path) -> dict:
    """Load final metrics from JSON file."""
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        return {}

    with open(metrics_path) as f:
        return json.load(f)


def plot_loss_curve(logs: list, output_path: Path) -> None:
    """Plot training loss curve."""
    train_logs = [l for l in logs if l["phase"] == "train"]
    eval_logs = [l for l in logs if l["phase"] == "eval"]

    if not train_logs:
        print("No training logs found, skipping loss curve")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Training loss
    train_steps = [l["step"] for l in train_logs]
    train_losses = [l["metrics"]["loss"] for l in train_logs]
    ax.plot(train_steps, train_losses, label="Train Loss", alpha=0.7)

    # Evaluation loss
    if eval_logs:
        eval_steps = [l["step"] for l in eval_logs]
        eval_losses = [l["metrics"]["loss"] for l in eval_logs]
        ax.plot(eval_steps, eval_losses, label="Val Loss", marker="o", markersize=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {output_path}")


def plot_accuracy_curve(logs: list, output_path: Path) -> None:
    """Plot training accuracy curve."""
    train_logs = [l for l in logs if l["phase"] == "train"]
    eval_logs = [l for l in logs if l["phase"] == "eval"]

    if not train_logs:
        print("No training logs found, skipping accuracy curve")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Training accuracy
    train_steps = [l["step"] for l in train_logs]
    train_accs = [l["metrics"]["accuracy"] for l in train_logs]
    ax.plot(train_steps, train_accs, label="Train Accuracy", alpha=0.7)

    # Evaluation accuracy
    if eval_logs:
        eval_steps = [l["step"] for l in eval_logs]
        eval_accs = [l["metrics"]["accuracy"] for l in eval_logs]
        ax.plot(eval_steps, eval_accs, label="Val Accuracy", marker="o", markersize=4)

    # Add 65% threshold line
    ax.axhline(y=0.65, color="r", linestyle="--", alpha=0.5, label="65% Target")

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved accuracy curve to {output_path}")


def plot_reward_distribution(logs: list, output_path: Path) -> None:
    """Plot reward distribution histogram from evaluation logs."""
    eval_logs = [l for l in logs if l["phase"] == "eval"]

    if not eval_logs:
        print("No evaluation logs found, skipping reward distribution")
        return

    # Use the last evaluation
    last_eval = eval_logs[-1]["metrics"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create synthetic distributions based on mean/std
    # (In a real scenario, we'd save raw rewards during evaluation)
    chosen_mean = last_eval.get("chosen_reward_mean", 0)
    chosen_std = last_eval.get("chosen_reward_std", 1)
    rejected_mean = last_eval.get("rejected_reward_mean", 0)
    rejected_std = last_eval.get("rejected_reward_std", 1)

    # Generate samples for visualization
    np.random.seed(42)
    chosen_samples = np.random.normal(chosen_mean, chosen_std, 1000)
    rejected_samples = np.random.normal(rejected_mean, rejected_std, 1000)

    ax.hist(chosen_samples, bins=50, alpha=0.6, label=f"Chosen (mean={chosen_mean:.2f})", color="green")
    ax.hist(rejected_samples, bins=50, alpha=0.6, label=f"Rejected (mean={rejected_mean:.2f})", color="red")

    ax.set_xlabel("Reward Score")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution (Chosen vs Rejected)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved reward distribution to {output_path}")


def plot_calibration(logs: list, output_path: Path) -> None:
    """Plot reward margin vs accuracy (calibration)."""
    eval_logs = [l for l in logs if l["phase"] == "eval"]

    if not eval_logs:
        print("No evaluation logs found, skipping calibration plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot reward margin over training
    steps = [l["step"] for l in eval_logs]
    margins = [l["metrics"].get("reward_margin", 0) for l in eval_logs]
    accuracies = [l["metrics"]["accuracy"] for l in eval_logs]

    # Create scatter plot with color by step
    scatter = ax.scatter(margins, accuracies, c=steps, cmap="viridis", s=50, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="Training Step")

    ax.set_xlabel("Reward Margin (chosen - rejected)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration: Reward Margin vs Accuracy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved calibration plot to {output_path}")


def plot_length_correlation(logs: list, output_path: Path) -> None:
    """Plot length-reward correlation over training."""
    eval_logs = [l for l in logs if l["phase"] == "eval"]

    if not eval_logs:
        print("No evaluation logs found, skipping length correlation plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    steps = [l["step"] for l in eval_logs]
    correlations = [l["metrics"].get("length_correlation", 0) for l in eval_logs]

    ax.plot(steps, correlations, marker="o", markersize=4)

    # Add threshold lines
    ax.axhline(y=0.3, color="r", linestyle="--", alpha=0.5, label="|r| = 0.3 threshold")
    ax.axhline(y=-0.3, color="r", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    ax.set_xlabel("Step")
    ax.set_ylabel("Length-Reward Correlation (r)")
    ax.set_title("Length-Reward Correlation Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved length correlation plot to {output_path}")


def plot_learning_rate(logs: list, output_path: Path) -> None:
    """Plot learning rate schedule."""
    train_logs = [l for l in logs if l["phase"] == "train"]

    if not train_logs:
        print("No training logs found, skipping learning rate plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    steps = [l["step"] for l in train_logs]
    lrs = [l["metrics"].get("learning_rate", 0) for l in train_logs]

    ax.plot(steps, lrs)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved learning rate plot to {output_path}")


def generate_all_plots(results_dir: str) -> None:
    """Generate all visualization plots."""
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print(f"Generating plots for {results_dir}")

    # Load logs
    try:
        logs = load_training_logs(results_path)
        print(f"Loaded {len(logs)} log entries")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        logs = []

    # Load final metrics
    final_metrics = load_final_metrics(results_path)
    if final_metrics:
        print("Final Metrics Summary:")
        if "best_val_accuracy" in final_metrics:
            print(f"  Best Val Accuracy: {final_metrics['best_val_accuracy']:.2%}")
        if "test" in final_metrics:
            print(f"  Test Accuracy: {final_metrics['test']['accuracy']:.2%}")
            print(f"  Length Correlation: {final_metrics['test']['length_correlation']:.3f}")

    if not logs:
        print("No logs to visualize. Run training first.")
        return

    # Create plots directory
    plots_dir = results_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate all plots
    plot_loss_curve(logs, plots_dir / "loss_curve.png")
    plot_accuracy_curve(logs, plots_dir / "accuracy_curve.png")
    plot_reward_distribution(logs, plots_dir / "reward_distribution.png")
    plot_calibration(logs, plots_dir / "calibration.png")
    plot_length_correlation(logs, plots_dir / "length_correlation.png")
    plot_learning_rate(logs, plots_dir / "learning_rate.png")

    print(f"\nAll plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize reward model training results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/m1_reward_model",
        help="Path to results directory",
    )
    args = parser.parse_args()

    generate_all_plots(args.results_dir)


if __name__ == "__main__":
    main()
