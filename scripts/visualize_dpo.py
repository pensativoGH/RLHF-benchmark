#!/usr/bin/env python3
"""Visualization script for DPO training results (Milestone 2)."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_training_history(results_dir: Path) -> list:
    """Load training history from JSON file."""
    history_path = results_dir / "logs" / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return []


def load_metrics(results_dir: Path) -> dict:
    """Load final metrics from JSON file."""
    metrics_path = results_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def load_win_rate(results_dir: Path) -> dict:
    """Load win rate metrics from JSON file."""
    win_rate_path = results_dir / "win_rate.json"
    if win_rate_path.exists():
        with open(win_rate_path) as f:
            return json.load(f)
    return {}


def plot_loss_curve(history: list, output_path: Path) -> None:
    """Plot DPO loss over training."""
    steps = [h["step"] for h in history if h.get("loss") is not None]
    losses = [h["loss"] for h in history if h.get("loss") is not None]

    if not steps:
        print("No loss data to plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, "b-", linewidth=1.5, alpha=0.8)

    # Add smoothed line
    if len(losses) > 10:
        window = min(20, len(losses) // 5)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        smoothed_steps = steps[window - 1:]
        plt.plot(smoothed_steps, smoothed, "r-", linewidth=2, label="Smoothed", alpha=0.8)
        plt.legend()

    plt.xlabel("Step")
    plt.ylabel("DPO Loss")
    plt.title("DPO Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_reward_margins(history: list, output_path: Path) -> None:
    """Plot reward margins (chosen - rejected) over training."""
    steps = [h["step"] for h in history if h.get("rewards/margins") is not None]
    margins = [h["rewards/margins"] for h in history if h.get("rewards/margins") is not None]

    if not steps:
        print("No reward margin data to plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, margins, "g-", linewidth=1.5, alpha=0.8)

    # Add smoothed line
    if len(margins) > 10:
        window = min(20, len(margins) // 5)
        smoothed = np.convolve(margins, np.ones(window) / window, mode="valid")
        smoothed_steps = steps[window - 1:]
        plt.plot(smoothed_steps, smoothed, "darkgreen", linewidth=2, label="Smoothed", alpha=0.8)
        plt.legend()

    plt.xlabel("Step")
    plt.ylabel("Reward Margin")
    plt.title("DPO Reward Margin (Chosen - Rejected)")
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_rewards_comparison(history: list, output_path: Path) -> None:
    """Plot chosen vs rejected rewards over training."""
    steps = [h["step"] for h in history if h.get("rewards/chosen") is not None]
    chosen_rewards = [h["rewards/chosen"] for h in history if h.get("rewards/chosen") is not None]
    rejected_rewards = [h["rewards/rejected"] for h in history if h.get("rewards/rejected") is not None]

    if not steps or len(chosen_rewards) != len(rejected_rewards):
        print("No reward comparison data to plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, chosen_rewards, "b-", linewidth=1.5, alpha=0.7, label="Chosen")
    plt.plot(steps, rejected_rewards, "r-", linewidth=1.5, alpha=0.7, label="Rejected")

    plt.xlabel("Step")
    plt.ylabel("Implicit Reward")
    plt.title("DPO Implicit Rewards: Chosen vs Rejected")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_learning_rate(history: list, output_path: Path) -> None:
    """Plot learning rate schedule over training."""
    steps = [h["step"] for h in history if h.get("learning_rate") is not None]
    lrs = [h["learning_rate"] for h in history if h.get("learning_rate") is not None]

    if not steps:
        print("No learning rate data to plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, "purple", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_win_rate_summary(win_rate: dict, output_path: Path) -> None:
    """Plot win rate summary as a bar chart."""
    if not win_rate:
        print("No win rate data to plot")
        return

    categories = ["Wins", "Losses", "Ties"]
    values = [
        win_rate.get("wins", 0),
        win_rate.get("losses", 0),
        win_rate.get("ties", 0),
    ]
    colors = ["green", "red", "gray"]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=12,
        )

    total = win_rate.get("total", sum(values))
    win_rate_pct = win_rate.get("win_rate", 0) * 100

    plt.title(f"Win Rate vs Base Model: {win_rate_pct:.1f}% (n={total})")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_kl_trajectory(win_rate: dict, output_path: Path) -> None:
    """Plot KL divergence information (placeholder - actual KL trajectory requires per-step logging)."""
    if not win_rate:
        print("No KL data to plot")
        return

    # Create a simple visualization of final KL divergence
    avg_kl = win_rate.get("avg_kl_divergence", 0)
    avg_margin = win_rate.get("avg_reward_margin", 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # KL divergence bar
    ax1.bar(["Average KL"], [avg_kl], color="orange", alpha=0.7, edgecolor="black")
    ax1.set_ylabel("KL Divergence (nats)")
    ax1.set_title("KL Divergence from Reference Model")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # Reward margin bar
    ax2.bar(["Average Margin"], [avg_margin], color="blue", alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Log Probability Difference")
    ax2.set_title("Average Reward Margin (Policy - Reference)")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_figure(history: list, win_rate: dict, output_path: Path) -> None:
    """Create a summary figure with all key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss curve
    ax1 = axes[0, 0]
    steps = [h["step"] for h in history if h.get("loss") is not None]
    losses = [h["loss"] for h in history if h.get("loss") is not None]
    if steps:
        ax1.plot(steps, losses, "b-", linewidth=1, alpha=0.6)
        if len(losses) > 10:
            window = min(20, len(losses) // 5)
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax1.plot(steps[window - 1:], smoothed, "r-", linewidth=2, label="Smoothed")
            ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("DPO Training Loss")
    ax1.grid(True, alpha=0.3)

    # 2. Reward margins
    ax2 = axes[0, 1]
    steps = [h["step"] for h in history if h.get("rewards/margins") is not None]
    margins = [h["rewards/margins"] for h in history if h.get("rewards/margins") is not None]
    if steps:
        ax2.plot(steps, margins, "g-", linewidth=1, alpha=0.6)
        if len(margins) > 10:
            window = min(20, len(margins) // 5)
            smoothed = np.convolve(margins, np.ones(window) / window, mode="valid")
            ax2.plot(steps[window - 1:], smoothed, "darkgreen", linewidth=2, label="Smoothed")
            ax2.legend()
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Margin")
    ax2.set_title("Reward Margin (Chosen - Rejected)")
    ax2.grid(True, alpha=0.3)

    # 3. Win rate
    ax3 = axes[1, 0]
    if win_rate:
        categories = ["Wins", "Losses", "Ties"]
        values = [win_rate.get("wins", 0), win_rate.get("losses", 0), win_rate.get("ties", 0)]
        colors = ["green", "red", "gray"]
        bars = ax3.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom")
        win_pct = win_rate.get("win_rate", 0) * 100
        ax3.set_title(f"Win Rate: {win_pct:.1f}%")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. KL and margin summary
    ax4 = axes[1, 1]
    if win_rate:
        metrics = ["Avg KL Div", "Avg Margin"]
        values = [win_rate.get("avg_kl_divergence", 0), win_rate.get("avg_reward_margin", 0)]
        colors = ["orange", "blue"]
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor="black")
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.3f}", ha="center", va="bottom")
        ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_title("Final Metrics")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle("DPO Training Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DPO training results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/m2_dpo",
        help="Directory containing DPO results",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    history = load_training_history(results_dir)
    metrics = load_metrics(results_dir)
    win_rate = load_win_rate(results_dir)

    print(f"Loaded {len(history)} training history entries")
    print(f"Metrics: {list(metrics.keys())}")
    print(f"Win rate data: {list(win_rate.keys())}")

    # Generate plots
    if history:
        plot_loss_curve(history, plots_dir / "loss_curve.png")
        plot_reward_margins(history, plots_dir / "reward_margins.png")
        plot_rewards_comparison(history, plots_dir / "rewards_comparison.png")
        plot_learning_rate(history, plots_dir / "learning_rate.png")

    if win_rate:
        plot_win_rate_summary(win_rate, plots_dir / "win_rate.png")
        plot_kl_trajectory(win_rate, plots_dir / "kl_trajectory.png")

    if history or win_rate:
        create_summary_figure(history, win_rate, plots_dir / "summary.png")

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
