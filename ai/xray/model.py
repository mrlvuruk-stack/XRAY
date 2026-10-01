"""
Chest X-Ray Model Architecture
Implements MONAI DenseNet121 for multi-finding chest radiograph interpretation.
Inherits from BaseMedicalModel for seamless registry integration.
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

from ..common.model_registry import BaseMedicalModel
from ..common.device import get_device
from .config import (
    MODEL_NAME,
    MODEL_VERSION,
    MODALITY,
    TASK,
    LABELS,
    DEFAULT_WEIGHTS_PATH,
    IN_CHANNELS,
    SPATIAL_DIMS,
)
from .preprocessing import XRayPreprocessor


class XRayDenseNetModel(BaseMedicalModel):
    """
    Chest X-ray deep neural network utilizing MONAI's DenseNet121 architecture.
    Operates strictly with real weights when configured, or signals unconfigured status.
    """

    def __init__(self, weights_path: Optional[str] = None):
        target_weights = weights_path or str(DEFAULT_WEIGHTS_PATH)
        super().__init__(
            name=MODEL_NAME,
            version=MODEL_VERSION,
            modality=MODALITY,
            task=TASK,
            labels=LABELS,
            weights_path=target_weights,
        )
        self.preprocessor = XRayPreprocessor()
        self.device = get_device()
        self.target_layer = "features.denseblock4.denselayer16.layers.conv2"

    def is_configured(self) -> bool:
        """Checks if real model weights exist at the configured path."""
        if not self.weights_path:
            return False
        return os.path.isfile(self.weights_path) and os.path.getsize(self.weights_path) > 0

    def load_weights(self, device: Optional[torch.device] = None) -> bool:
        """
        Instantiates MONAI DenseNet121 and loads trained weights if present.
        Never fabricates weights.
        """
        if device is not None:
            self.device = device

        if not self.is_configured():
            self._model = None
            self._is_loaded = False
            return False

        try:
            model = DenseNet121(
                spatial_dims=SPATIAL_DIMS,
                in_channels=IN_CHANNELS,
                out_channels=len(self.labels),
            )
            state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            # Handle both full checkpoint or pure state_dict
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self._model = model
            self._is_loaded = True
            return True
        except Exception as e:
            self._model = None
            self._is_loaded = False
            return False

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """Preprocesses input radiograph using MONAI transforms."""
        return self.preprocessor.preprocess(pil_image)

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Executes model inference on a normalized 4D tensor (1, 1, 224, 224).
        Returns multi-label probabilities via sigmoid activation.
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Chest X-ray model is not configured. Weights not loaded.")

        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            logits = self._model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        findings = []
        for idx, label in enumerate(self.labels):
            prob = float(probs[idx])
            findings.append({
                "finding": label,
                "probability": round(prob, 4),
                "status": "High" if prob >= 0.50 else ("Moderate" if prob >= 0.25 else "Low"),
            })

        # Sort findings by probability descending
        findings.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "findings": findings,
            "raw_probabilities": probs.tolist(),
            "top_finding": findings[0]["finding"],
            "top_probability": findings[0]["probability"],
        }

    def explain(
        self,
        pil_image: Image.Image,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generates Grad-CAM visual explanation using MONAI GradCAM."""
        from .explainability import compute_xray_gradcam
        return compute_xray_gradcam(
            model=self._model,
            target_layer=self.target_layer,
            input_tensor=input_tensor,
            pil_image=pil_image,
            target_class=target_class,
            device=self.device,
        )
