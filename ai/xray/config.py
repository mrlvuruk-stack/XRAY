"""
Chest X-Ray Configuration Module
Defines target pathology classes, image resolution, thresholds, and weights paths.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "xray"
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "chexnet_monai_densenet121.pth"

# Model Architecture Configuration
MODEL_NAME = "Swasthya-MONAI-CheXNet-DenseNet121"
MODEL_VERSION = "v1.2.0"
MODALITY = "xray"
TASK = "classification"

# Radiographic Findings / Pathologies
LABELS = [
    "Normal",
    "Pneumonia",
    "Pleural Effusion",
    "Atelectasis",
    "Cardiomegaly",
    "Infiltration",
    "Consolidation",
    "Nodule",
]

# Image Preprocessing Specs
INPUT_SIZE = (224, 224)
IN_CHANNELS = 1
SPATIAL_DIMS = 2

# Clinical Decision Support Thresholds
ALERT_THRESHOLD_HIGH = 0.50
ALERT_THRESHOLD_MODERATE = 0.25

# Weights Setup Guide
WEIGHTS_SETUP_INSTRUCTIONS = """
To configure the real Chest X-Ray AI model:
1. Download a compatible MONAI DenseNet121 or CheXNet pretrained checkpoint (.pth).
2. Save the file to: `models/xray/chexnet_monai_densenet121.pth`
3. Restart or reload the model from Settings / Navigation.
4. Ensure the state dict matches DenseNet121 with 8 output classes.
"""
