"""Constants for class labels used across the project."""

NUM_DR_CLASSES = 5
NUM_EDEMA_CLASSES = 3

DR_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "PDR"}
EDEMA_LABELS = {0: "No Edema", 1: "Possible/Mild Edema", 2: "Clinically Significant Edema"}

DR_COLUMN = "Retinopathy grade"
EDEMA_COLUMN = "Risk of macular edema"
IMAGE_NAME_COLUMN = "Image name"
