"""Helper functions for the project."""

import logging

import torch

from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("helper")


def get_device(device: str = "auto") -> torch.device:
    """Get the device to use for inference."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_metrics(metrics: dict, splits: tuple[str, str] = ("train", "val")) -> None:
    """Log training and validation metrics in a clean tabular format."""

    headers = ["Split", "Loss", "Acc", "F1", "Prec", "Recall", "QWK"]

    def row(split: str):
        return [
            split.capitalize(),
            f"{metrics.get(f'{split}_loss', 0.0):.4f}",
            f"{metrics.get(f'{split}_accuracy', 0.0):.4f}",
            f"{metrics.get(f'{split}_f1', 0.0):.4f}",
            f"{metrics.get(f'{split}_precision', 0.0):.4f}",
            f"{metrics.get(f'{split}_recall', 0.0):.4f}",
            f"{metrics.get(f'{split}_qwk', 0.0):.4f}",
        ]

    rows = [row(s) for s in splits]

    fmt = "{:<8} | {:<8} {:<6} {:<6} {:<6} {:<7} {:<6}"
    if "epoch" in metrics and "learning_rate" in metrics:
        LOGGER.info(f"\nEpoch {metrics['epoch']} | LR: {metrics['learning_rate']:.6f}")
    LOGGER.info("-" * 78)
    LOGGER.info(fmt.format(*headers))
    LOGGER.info("-" * 78)
    for r in rows:
        LOGGER.info(fmt.format(*r))
    LOGGER.info("-" * 78)
