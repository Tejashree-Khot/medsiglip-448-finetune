"""Dataset and dataloader for the IDRiD retinal disease grading task."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoProcessor

from dataloader.class_labels import DR_COLUMN, EDEMA_COLUMN, IMAGE_NAME_COLUMN
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("data_loader")


def load_annotations(dataset_path: Path) -> pd.DataFrame:
    """Load and validate annotations CSV from the dataset directory."""
    csv_path = dataset_path / "annotations.csv"
    df = pd.read_csv(csv_path)
    LOGGER.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


class IDRiDDataset(Dataset):
    """IDRiD dataset returning pixel values with DR and Edema labels."""

    def __init__(self, processor: AutoProcessor, dataset_path: Path):
        self.processor = processor
        self.dataset_path = dataset_path

        df = load_annotations(dataset_path)
        self.image_names: list[str] = df[IMAGE_NAME_COLUMN].tolist()
        self.dr_labels: list[int] = df[DR_COLUMN].astype(int).tolist()
        self.edema_labels: list[int] = df[EDEMA_COLUMN].astype(int).tolist()

        LOGGER.info(f"Dataset ready: {len(self)} samples from {dataset_path}")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> dict:
        image_path = self.dataset_path / "images" / self.image_names[index]
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "dr_labels": self.dr_labels[index],
            "edema_labels": self.edema_labels[index],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate batch into stacked tensors."""
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "dr_labels": torch.tensor([item["dr_labels"] for item in batch], dtype=torch.long),
        "edema_labels": torch.tensor([item["edema_labels"] for item in batch], dtype=torch.long),
    }


def get_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    """Get weighted random sampler for class balancing."""
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_data_loader(dataset: IDRiDDataset, batch_size: int, use_weighted_sampler: bool = False) -> DataLoader:
    """Build a DataLoader with optional weighted sampling on DR labels."""
    from dataloader.class_labels import NUM_DR_CLASSES

    sampler = get_sampler(dataset.dr_labels, NUM_DR_CLASSES) if use_weighted_sampler else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
