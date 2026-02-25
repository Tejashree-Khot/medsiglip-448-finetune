"""Reusable model evaluation utilities."""

import logging

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("evaluate")


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device, criterion: nn.Module, data_type: str = "val"
) -> dict[str, float]:
    """Evaluate model on a dataloader and return metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "f1": f"{current_f1:.4f}"})
    metrics = {
        f"{data_type}_loss": total_loss / len(dataloader),
        f"{data_type}_accuracy": accuracy_score(all_labels, all_preds),
        f"{data_type}_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        f"{data_type}_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        f"{data_type}_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        f"{data_type}_qwk": cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
    }

    if data_type == "test":
        df = pd.DataFrame({"predictions": all_preds, "labels": all_labels})
        df.to_csv("test_predictions.csv", index=False)

    return metrics
