"""Main trainer class for dual-head MedSigLIP model."""

import logging

import wandb
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from tqdm import tqdm
from transformers import AutoProcessor

from dataloader.data_loader import IDRiDDataset, get_data_loader
from models.checkpoint import CheckpointManager
from models.model import DualHeadMedSigLIP
from trainer.config import TrainerConfig
from utils.evaluate import evaluate_model
from utils.helper import get_device, log_metrics
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("trainer")


class Trainer:
    """Trainer class for dual-head model training with validation and scheduling."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = get_device(self.config.device)
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.train_loader, self.val_loader = self._setup_dataloaders()
        self.best_val_loss = float("inf")

    def _setup_model(self) -> DualHeadMedSigLIP:
        """Setup and return the dual-head model."""
        model = DualHeadMedSigLIP(
            self.config.model_id,
            alpha=self.config.alpha,
            beta=self.config.beta,
            freeze_backbone=self.config.freeze_backbone,
        )
        model.to(self.device)
        LOGGER.info(
            f"Model loaded: {self.config.model_id} | "
            f"Total params: {model.get_num_parameters():,} | "
            f"Trainable: {model.get_num_parameters(trainable_only=True):,}"
        )
        return model

    def _setup_optimizer(self) -> AdamW | SGD:
        """Setup and return the optimizer."""
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.config.optimizer == "sgd":
            return SGD(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        return AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _setup_scheduler(self) -> LRScheduler | None:
        """Setup and return the learning rate scheduler."""
        if self.config.scheduler == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        return None

    def _setup_dataloaders(self) -> tuple:
        """Setup and return train and validation dataloaders."""
        processor = AutoProcessor.from_pretrained(self.config.model_id)

        train_dataset = IDRiDDataset(processor=processor, dataset_path=self.config.train_path)
        val_dataset = IDRiDDataset(processor=processor, dataset_path=self.config.val_path)

        train_loader = get_data_loader(
            train_dataset, batch_size=self.config.batch_size, use_weighted_sampler=self.config.use_weighted_sampler
        )
        val_loader = get_data_loader(val_dataset, batch_size=self.config.batch_size)

        LOGGER.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        return train_loader, val_loader

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch and return average losses."""
        self.model.train()
        total_loss, total_dr_loss, total_edema_loss = 0.0, 0.0, 0.0
        dr_correct, edema_correct, total = 0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs} [Train]")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(self.device)
            dr_labels = batch["dr_labels"].to(self.device)
            edema_labels = batch["edema_labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(pixel_values=pixel_values, dr_labels=dr_labels, edema_labels=edema_labels)
            outputs.loss.backward()
            self.optimizer.step()

            total_loss += outputs.loss.item()
            total_dr_loss += outputs.dr_loss.item()
            total_edema_loss += outputs.edema_loss.item()

            dr_correct += (outputs.dr_logits.argmax(dim=1) == dr_labels).sum().item()
            edema_correct += (outputs.edema_logits.argmax(dim=1) == edema_labels).sum().item()
            total += dr_labels.size(0)

            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        n = len(self.train_loader)
        return {
            "train_loss": total_loss / n,
            "train_dr_loss": total_dr_loss / n,
            "train_edema_loss": total_edema_loss / n,
            "train_dr_acc": dr_correct / total,
            "train_edema_acc": edema_correct / total,
        }

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False) -> None:
        """Save model checkpoint."""
        suffix = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = self.config.checkpoint_dir / suffix
        CheckpointManager.save_checkpoint(self.model, self.optimizer, epoch, metrics, path)

    def train(self) -> dict:
        """Main training loop."""
        LOGGER.info(f"Starting training on {self.device}")

        if self.config.freeze_backbone:
            self.model.freeze_backbone()

        if self.config.resume_checkpoint_path:
            self.model, epoch, metrics = CheckpointManager.load_checkpoint(
                self.model, self.config.resume_checkpoint_path, device=self.device, optimizer=self.optimizer
            )
            LOGGER.info(f"Resumed training from epoch {epoch}")

        for epoch in range(1, self.config.epochs + 1):
            if epoch == self.config.unfreeze_epoch and self.config.unfreeze_all:
                self.optimizer.param_groups[0]["lr"] /= 10
                self.model.unfreeze_all()
                LOGGER.info(f"Unfroze all layers at epoch {epoch}")

            train_metrics = self._train_epoch(epoch)
            val_metrics = evaluate_model(self.model, self.val_loader, self.device)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            metrics = {**train_metrics, **val_metrics, "learning_rate": current_lr, "epoch": epoch}
            log_metrics(metrics)
            wandb.log(metrics)

            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self._save_checkpoint(epoch, val_metrics, is_best=True)

        wandb.finish()
        return {"best_val_loss": self.best_val_loss}
