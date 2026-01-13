"""Unified logging with JSON Lines support for training metrics."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up Python logging with console output."""
    logger = logging.getLogger("rlcode")
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class MetricsLogger:
    """Logger for training metrics in JSON Lines format."""

    def __init__(self, log_dir: str, filename: str = "train_metrics.jsonl"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / filename
        self.logger = setup_logging()

        # Initialize log file
        if not self.log_path.exists():
            self.log_path.touch()

    def log(
        self,
        step: int,
        metrics: Dict[str, Any],
        phase: str = "train",
        epoch: Optional[int] = None,
    ) -> None:
        """Log metrics to JSON Lines file and console."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "phase": phase,
            "metrics": metrics,
        }
        if epoch is not None:
            record["epoch"] = epoch

        # Write to JSON Lines file
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Log to console
        metrics_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items())
        prefix = f"[{phase.upper()}]"
        if epoch is not None:
            prefix += f" Epoch {epoch}"
        self.logger.info(f"{prefix} Step {step} | {metrics_str}")

    def log_eval(self, step: int, metrics: Dict[str, Any], epoch: Optional[int] = None) -> None:
        """Log evaluation metrics."""
        self.log(step, metrics, phase="eval", epoch=epoch)

    def save_final_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> None:
        """Save final metrics to a JSON file."""
        output_path = self.log_dir.parent / filename
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Final metrics saved to {output_path}")

    def load_metrics(self) -> list:
        """Load all logged metrics from JSON Lines file."""
        metrics = []
        if self.log_path.exists():
            with open(self.log_path) as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
        return metrics
