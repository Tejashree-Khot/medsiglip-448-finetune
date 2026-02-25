"""Download the IDRiD dataset from Kaggle."""

import shutil
from pathlib import Path

import kagglehub

TARGET_DIR = Path(__file__).parent.parent / "data"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Download to kagglehub cache
cache_path = Path(kagglehub.dataset_download("mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid"))

# Copy dataset to target directory
shutil.copytree(cache_path, TARGET_DIR, dirs_exist_ok=True)
print("Path to dataset files:", TARGET_DIR)
