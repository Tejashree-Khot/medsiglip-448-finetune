"""Checkpoint manager for saving and loading model weights."""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manager class for saving and loading model checkpoints."""

    @staticmethod
    def save_checkpoint(
        model: nn.Module, optimizer: Optimizer | None, epoch: int, metrics: dict[str, Any], path: str | Path
    ) -> None:
        """Save a training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @staticmethod
    def load_checkpoint(
        model: nn.Module, path: str | Path, device: str | torch.device = "cpu", optimizer: Optimizer | None = None
    ) -> tuple[nn.Module, int, dict[str, Any]]:
        """Load a checkpoint for resuming training."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Move optimizer state to correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})

        logger.info(f"Checkpoint loaded from {path} (epoch {epoch})")
        return model, epoch, metrics

    @staticmethod
    def load_for_inference(model: nn.Module, path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
        """Load model weights for inference only."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Handle both full checkpoints and state_dict only files
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        logger.info(f"Model loaded for inference from {path}")
        return model

    @staticmethod
    def load_pretrained_weights(
        model: nn.Module, path: str | Path, device: str | torch.device = "cpu", strict: bool = True
    ) -> nn.Module:
        """Load pretrained weights with flexible matching."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Handle both full checkpoints and state_dict only files
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if strict:
            model.load_state_dict(state_dict)
        else:
            # Load with partial matching
            model_dict = model.state_dict()
            # Filter out mismatched keys
            pretrained_dict = {
                k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape
            }
            skipped = set(state_dict.keys()) - set(pretrained_dict.keys())
            if skipped:
                logger.warning(f"Skipped loading {len(skipped)} layers: {skipped}")

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        model.to(device)
        logger.info(f"Pretrained weights loaded from {path}")
        return model
