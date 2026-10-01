"""
Ultrasound Configuration Module
Configures classification and segmentation pipelines for sonography.
Separate from Chest X-Ray pipeline.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "ultrasound"
CLS_WEIGHTS_PATH = MODELS_DIR / "classification" / "ultrasound_densenet121.pth"
SEG_WEIGHTS_PATH = MODELS_DIR / "segmentation" / "ultrasound_unet.pth"

# Model Identification
MODALITY = "ultrasound"
CLS_MODEL_NAME = "Swasthya-MONAI-Ultrasound-Classifier"
SEG_MODEL_NAME = "Swasthya-MONAI-Ultrasound-UNet"
MODEL_VERSION = "v1.1.0"

# Ultrasound Classification Targets
CLS_LABELS = ["Normal", "Benign Lesion", "Malignant Finding"]

# Ultrasound Segmentation Targets
SEG_LABELS = ["Background", "Lesion / Anomaly"]

# Image Input Dimensions
INPUT_SIZE = (256, 256)
IN_CHANNELS = 1
SPATIAL_DIMS = 2

# Weights Configuration Guide
WEIGHTS_SETUP_INSTRUCTIONS = {
    "classification": """
To configure the real Ultrasound Classification model:
1. Download a trained MONAI DenseNet121 sonography checkpoint.
2. Save to: `models/ultrasound/classification/ultrasound_densenet121.pth`
3. Ensure checkpoint has 3 output classes (Normal, Benign, Malignant).
""",
    "segmentation": """
To configure the real Ultrasound Segmentation model:
1. Download a trained MONAI UNet sonography segmentation checkpoint.
2. Save to: `models/ultrasound/segmentation/ultrasound_unet.pth`
3. Ensure network output matches binary lesion mask.
"""
}
