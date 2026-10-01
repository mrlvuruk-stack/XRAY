"""
Ultrasound Model Architectures
Implements separate MONAI DenseNet121 for classification and MONAI UNet for segmentation.
Completely decoupled from the Chest X-Ray pipeline.
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, UNet

from ..common.model_registry import BaseMedicalModel
from ..common.device import get_device
from .config import (
    MODALITY,
    CLS_MODEL_NAME,
    SEG_MODEL_NAME,
    MODEL_VERSION,
    CLS_LABELS,
    SEG_LABELS,
    CLS_WEIGHTS_PATH,
    SEG_WEIGHTS_PATH,
    IN_CHANNELS,
    SPATIAL_DIMS,
)
from .preprocessing import UltrasoundPreprocessor


class UltrasoundClassificationModel(BaseMedicalModel):
    """
    Ultrasound pathology classification model utilizing MONAI DenseNet121.
    Distinguishes between Normal, Benign Lesion, and Malignant Finding.
    """

    def __init__(self, weights_path: Optional[str] = None):
        target_path = weights_path or str(CLS_WEIGHTS_PATH)
        super().__init__(
            name=CLS_MODEL_NAME,
            version=MODEL_VERSION,
            modality=MODALITY,
            task="classification",
            labels=CLS_LABELS,
            weights_path=target_path,
        )
        self.preprocessor = UltrasoundPreprocessor()
        self.device = get_device()
        self.target_layer = "features.denseblock4.denselayer16.layers.conv2"

    def is_configured(self) -> bool:
        """Returns True if classification weights exist on disk."""
        if not self.weights_path:
            return False
        return os.path.isfile(self.weights_path) and os.path.getsize(self.weights_path) > 0

    def load_weights(self, device: Optional[torch.device] = None) -> bool:
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
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self._model = model
            self._is_loaded = True
            return True
        except Exception:
            self._model = None
            self._is_loaded = False
            return False

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        return self.preprocessor.preprocess(pil_image)

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Ultrasound classification model is not configured.")

        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            logits = self._model(input_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        findings = []
        for idx, label in enumerate(self.labels):
            findings.append({
                "finding": label,
                "probability": round(float(probs[idx]), 4),
                "status": "High" if probs[idx] >= 0.5 else ("Moderate" if probs[idx] >= 0.25 else "Low"),
            })

        findings.sort(key=lambda x: x["probability"], reverse=True)
        return {
            "findings": findings,
            "top_finding": findings[0]["finding"],
            "top_probability": findings[0]["probability"],
            "raw_probabilities": probs.tolist(),
        }

    def explain(
        self,
        pil_image: Image.Image,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        from .explainability import compute_ultrasound_gradcam
        return compute_ultrasound_gradcam(
            model=self._model,
            target_layer=self.target_layer,
            input_tensor=input_tensor,
            pil_image=pil_image,
            target_class=target_class,
            device=self.device,
        )


class UltrasoundSegmentationModel(BaseMedicalModel):
    """
    Ultrasound lesion segmentation model utilizing MONAI UNet architecture.
    Produces lesion boundary mask and area quantification.
    """

    def __init__(self, weights_path: Optional[str] = None):
        target_path = weights_path or str(SEG_WEIGHTS_PATH)
        super().__init__(
            name=SEG_MODEL_NAME,
            version=MODEL_VERSION,
            modality=MODALITY,
            task="segmentation",
            labels=SEG_LABELS,
            weights_path=target_path,
        )
        self.preprocessor = UltrasoundPreprocessor()
        self.device = get_device()

    def is_configured(self) -> bool:
        """Returns True if segmentation weights exist on disk."""
        if not self.weights_path:
            return False
        return os.path.isfile(self.weights_path) and os.path.getsize(self.weights_path) > 0

    def load_weights(self, device: Optional[torch.device] = None) -> bool:
        if device is not None:
            self.device = device

        if not self.is_configured():
            self._model = None
            self._is_loaded = False
            return False

        try:
            model = UNet(
                spatial_dims=SPATIAL_DIMS,
                in_channels=IN_CHANNELS,
                out_channels=1,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2),
            )
            state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self._model = model
            self._is_loaded = True
            return True
        except Exception:
            self._model = None
            self._is_loaded = False
            return False

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        return self.preprocessor.preprocess(pil_image)

    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Ultrasound segmentation model is not configured.")

        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            logits = self._model(input_tensor)
            # Sigmoid for binary lesion segmentation
            pred_mask = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary_mask = (pred_mask >= 0.5).astype(np.uint8)
        lesion_pixel_count = int(np.sum(binary_mask))
        total_pixels = binary_mask.size
        lesion_ratio = round((lesion_pixel_count / total_pixels) * 100, 2)

        return {
            "mask_prob": pred_mask,
            "binary_mask": binary_mask,
            "lesion_pixels": lesion_pixel_count,
            "lesion_area_percentage": lesion_ratio,
        }

    def explain(
        self,
        pil_image: Image.Image,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        from .explainability import create_segmentation_overlay
        pred = self.predict(input_tensor)
        return create_segmentation_overlay(pil_image, pred["binary_mask"])
