"""Entry point for dual-head MedSigLIP training."""

import argparse
import logging
import os
from pathlib import Path

import wandb

from trainer.config import TrainerConfig
from trainer.trainer import Trainer
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("train")

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def make_argparser() -> argparse.ArgumentParser:
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(description="Train dual-head MedSigLIP model.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run dual-head MedSigLIP training."""
    root = Path(__file__).parent.parent

    config = TrainerConfig(
        model_id="google/medsiglip-448",
        train_path=root / "data" / "IDRiD" / "Train",
        val_path=root / "data" / "IDRiD" / "Test",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        scheduler=args.scheduler,
        optimizer=args.optimizer,
        freeze_backbone=args.freeze_backbone,
        unfreeze_all=args.unfreeze_all,
        unfreeze_epoch=args.unfreeze_epoch,
        wandb_project=args.wandb_project,
        resume_checkpoint_path=Path(args.resume) if args.resume else None,
    )

    wandb.init(project=config.wandb_project, config=vars(config))

    LOGGER.info("Training config: %s", config)
    trainer = Trainer(config)
    result = trainer.train()
    LOGGER.info(f"Training finished. Best val loss: {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    main(make_argparser().parse_args())
