"""Evaluation script for the dual-head MedSigLIP classification model."""

import argparse
import logging
from pathlib import Path

from transformers import AutoProcessor

from dataloader.data_loader import IDRiDDataset, get_data_loader
from models.checkpoint import CheckpointManager
from models.model import DualHeadMedSigLIP
from utils.evaluate import evaluate_model
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("test")

MODEL_ID = "google/medsiglip-448"


def make_argparser() -> argparse.ArgumentParser:
    """Create argument parser for test script."""
    parser = argparse.ArgumentParser(description="Evaluate dual-head MedSigLIP model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser


def main(args: argparse.Namespace) -> None:
    """Run evaluation on the test dataset."""
    root = Path(__file__).parent.parent
    device = 

    LOGGER.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = DualHeadMedSigLIP(MODEL_ID)
    model = CheckpointManager.load_for_inference(model, args.checkpoint, device=device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    test_path = root / "data" / "IDRiD" / "Test"

    LOGGER.info(f"Loading test data from: {test_path}")
    test_dataset = IDRiDDataset(processor=processor, dataset_path=test_path)
    test_loader = get_data_loader(test_dataset, batch_size=args.batch_size)
    LOGGER.info(f"Test samples: {len(test_dataset)}")

    LOGGER.info("Running evaluation...")
    metrics = evaluate_model(model, test_loader, device, data_type="test")

    LOGGER.info("--- Diabetic Retinopathy ---")
    LOGGER.info(
        f"Accuracy: {metrics['test_dr_acc']:.4f} | F1: {metrics['test_dr_f1']:.4f} | QWK: {metrics['test_dr_qwk']:.4f}"
    )

    LOGGER.info("--- Macular Edema ---")
    LOGGER.info(
        f"Accuracy: {metrics['test_edema_acc']:.4f} | F1: {metrics['test_edema_f1']:.4f} | QWK: {metrics['test_edema_qwk']:.4f}"
    )


if __name__ == "__main__":
    main(make_argparser().parse_args())
