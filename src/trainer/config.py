"""Training configuration dataclass."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainerConfig:
    """Configuration for dual-head model training."""

    model_id: str = "google/medsiglip-448"
    train_path: Path = Path("")
    val_path: Path = Path("")
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    alpha: float = 1.0
    beta: float = 1.0
    scheduler: str = "cosine"
    optimizer: str = "adam"
    freeze_backbone: bool = False
    unfreeze_all: bool = True
    unfreeze_epoch: int = 0
    checkpoint_dir: Path = Path(__file__).parent.parent.parent / "output" / "checkpoints"
    wandb_project: str = "medsiglip-dual-head"
    device: str = "auto"
    use_weighted_sampler: bool = True
    resume_checkpoint_path: Path | None = None
