"""
Chest X-Ray Pipeline Engine
Coordinates MONAI preprocessing, DenseNet121 inference, and Grad-CAM visualization.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import torch
from monai.networks.nets import DenseNet121

from .preprocessing import MedicalPreprocessor
from .explainability import compute_gradcam, generate_demo_xray_cam

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "xray" / "chexnet_monai_densenet121.pth"

MODEL_NAME = "Swasthya-MONAI-DenseNet121"
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


class XRayEngine:
    """
    Singleton-style engine for Chest X-Ray inference. Loads model once on CPU/CUDA.
    """
    _model: Optional[DenseNet121] = None
    _is_configured: bool = False

    def __init__(self):
        self.preprocessor = MedicalPreprocessor(spatial_size=(224, 224))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._check_configuration()

    def _check_configuration(self):
        self._is_configured = WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 0

    def is_configured(self) -> bool:
        self._check_configuration()
        return self._is_configured

    def _load_model(self):
        if self._model is None and self.is_configured():
            try:
                model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=len(LABELS))
                weights = torch.load(str(WEIGHTS_PATH), map_location=self.device, weights_only=True)
                if isinstance(weights, dict) and "state_dict" in weights:
                    weights = weights["state_dict"]
                model.load_state_dict(weights, strict=False)
                model.to(self.device)
                model.eval()
                self._model = model
            except Exception:
                self._model = None

    def analyze(self, pil_image: Image.Image, demo_mode: bool = True) -> Dict[str, Any]:
        """
        Runs the full diagnostic flow with timing and MONAI telemetry.
        """
        start_time = time.time()

        # Step 1: MONAI Preprocessing
        input_tensor, telemetry = self.preprocessor.process(pil_image)

        # Check real model availability
        if self.is_configured():
            self._load_model()
            if self._model is not None:
                with torch.inference_mode():
                    logits = self._model(input_tensor.to(self.device))
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

                findings = []
                for i, label in enumerate(LABELS):
                    p = float(probs[i])
                    findings.append({
                        "finding": label,
                        "probability": round(p, 4),
                        "status": "HIGH" if p >= 0.50 else ("MODERATE" if p >= 0.25 else "LOW")
                    })
                findings.sort(key=lambda x: x["probability"], reverse=True)

                # Grad-CAM on top finding
                top_idx = 0
                for i, l in enumerate(LABELS):
                    if l == findings[0]["finding"]:
                        top_idx = i
                        break

                images = compute_gradcam(
                    model=self._model,
                    target_layer="features.denseblock4.denselayer16.layers.conv2",
                    input_tensor=input_tensor,
                    pil_image=pil_image,
                    class_idx=top_idx
                )

                elapsed = round(time.time() - start_time, 2)
                return {
                    "status": "REAL_MODEL",
                    "is_demo": False,
                    "model_name": MODEL_NAME,
                    "inference_time_sec": elapsed,
                    "findings": findings,
                    "images": images,
                    "telemetry": telemetry,
                    "top_finding": findings[0]["finding"],
                    "top_probability": findings[0]["probability"],
                    "message": "Real MONAI DenseNet inference completed."
                }

        # When no trained model is configured:
        # Strictly enforce clinical integrity — never fabricate medical findings or predictions.
        elapsed = round(time.time() - start_time, 2)
        preprocessed_img = self.preprocessor.tensor_to_pil(input_tensor)

        return {
            "status": "MODEL_NOT_CONFIGURED",
            "is_demo": True,
            "is_configured": False,
            "model_name": f"{MODEL_NAME} (Preprocessing Only)",
            "inference_time_sec": elapsed,
            "findings": [],
            "images": {
                "original": pil_image,
                "preprocessed": preprocessed_img,
                "heatmap": None,
                "overlay": None,
            },
            "telemetry": telemetry,
            "top_finding": "No Model Configured",
            "top_probability": 0.0,
            "message": "Model not configured — preprocessing/demo pipeline only.",
            "instructions": f"Place validated DenseNet121 weights at: {WEIGHTS_PATH}"
        }


