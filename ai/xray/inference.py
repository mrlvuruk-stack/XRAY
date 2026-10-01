"""
Chest X-Ray Inference Pipeline
Executes validation, MONAI preprocessing, neural network inference, Grad-CAM generation,
and strictly enforces REAL MODEL vs DEMO MODE vs MODEL NOT CONFIGURED states.
"""

import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from ..common.device import get_device, get_device_info
from ..common.image_validation import validate_medical_image, ValidationResult
from .config import (
    MODEL_NAME,
    MODEL_VERSION,
    LABELS,
    DEFAULT_WEIGHTS_PATH,
    WEIGHTS_SETUP_INSTRUCTIONS,
)
from .preprocessing import XRayPreprocessor
from .model import XRayDenseNetModel
from .explainability import generate_demo_gradcam


@dataclass
class XRayResult:
    """Standardized result output for Chest X-Ray pipeline."""
    status: str  # "REAL_MODEL" | "DEMO_MODE" | "MODEL_NOT_CONFIGURED" | "ERROR"
    is_demo: bool
    is_configured: bool
    findings: List[Dict[str, Any]]
    top_finding: str
    top_probability: float
    original_image: Optional[Image.Image]
    heatmap: Optional[Image.Image]
    overlay: Optional[Image.Image]
    model_name: str
    model_version: str
    device_name: str
    inference_time_ms: float
    timestamp: str
    message: Optional[str] = None
    instructions: Optional[str] = None
    validation_meta: Optional[Dict[str, Any]] = None


class XRayPipeline:
    """
    Complete Chest X-ray processing pipeline.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path or str(DEFAULT_WEIGHTS_PATH)
        self.device = get_device()
        self.model = XRayDenseNetModel(weights_path=self.weights_path)
        self.preprocessor = XRayPreprocessor()
        self._cached_loaded = False

    def is_model_configured(self) -> bool:
        """Returns True if trained weights exist."""
        return self.model.is_configured()

    def run(
        self,
        image_input: Any,
        filename: str = "xray_scan.png",
        demo_mode: bool = False,
    ) -> XRayResult:
        """
        Executes end-to-end chest X-ray processing.
        Handles real inference, demo simulation, and unconfigured detection.
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        device_info = get_device_info()
        device_str = device_info["device_str"]

        # Step 1: Validation
        val_res: ValidationResult = validate_medical_image(
            file_bytes_or_path=image_input,
            filename=filename,
            modality="xray"
        )
        if not val_res.is_valid:
            return XRayResult(
                status="ERROR",
                is_demo=demo_mode,
                is_configured=self.is_model_configured(),
                findings=[],
                top_finding="N/A",
                top_probability=0.0,
                original_image=None,
                heatmap=None,
                overlay=None,
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                device_name=device_str,
                inference_time_ms=0.0,
                timestamp=timestamp,
                message=val_res.error_message,
                validation_meta=None
            )

        pil_img = val_res.pil_image

        # Step 2: Check Model Configuration
        is_configured = self.is_model_configured()

        # CASE A: Real Model Configured with Weights
        if is_configured:
            if not self._cached_loaded:
                self.model.load_weights(self.device)
                self._cached_loaded = True

            try:
                # MONAI Preprocessing
                input_tensor = self.preprocessor.preprocess(pil_img)

                # Prediction
                pred_data = self.model.predict(input_tensor)

                # Explainability: Grad-CAM on highest finding or class 1
                top_class_idx = 0
                for idx, lbl in enumerate(self.model.labels):
                    if lbl == pred_data["top_finding"]:
                        top_class_idx = idx
                        break

                cam_data = self.model.explain(
                    pil_image=pil_img,
                    input_tensor=input_tensor,
                    target_class=top_class_idx,
                )

                elapsed_ms = round((time.time() - start_time) * 1000, 2)
                return XRayResult(
                    status="REAL_MODEL",
                    is_demo=False,
                    is_configured=True,
                    findings=pred_data["findings"],
                    top_finding=pred_data["top_finding"],
                    top_probability=pred_data["top_probability"],
                    original_image=cam_data["images"]["original"],
                    heatmap=cam_data["images"]["heatmap"],
                    overlay=cam_data["images"]["overlay"],
                    model_name=MODEL_NAME,
                    model_version=MODEL_VERSION,
                    device_name=device_str,
                    inference_time_ms=elapsed_ms,
                    timestamp=timestamp,
                    message="Analysis completed using real pretrained weights.",
                    validation_meta=val_res.metadata
                )
            except Exception as e:
                elapsed_ms = round((time.time() - start_time) * 1000, 2)
                return XRayResult(
                    status="ERROR",
                    is_demo=False,
                    is_configured=True,
                    findings=[],
                    top_finding="N/A",
                    top_probability=0.0,
                    original_image=pil_img,
                    heatmap=None,
                    overlay=None,
                    model_name=MODEL_NAME,
                    model_version=MODEL_VERSION,
                    device_name=device_str,
                    inference_time_ms=elapsed_ms,
                    timestamp=timestamp,
                    message=f"Inference error encountered during execution.",
                    validation_meta=val_res.metadata
                )

        # CASE B: Model Weights Not Found, Demo Mode is Disabled
        if not demo_mode:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return XRayResult(
                status="MODEL_NOT_CONFIGURED",
                is_demo=False,
                is_configured=False,
                findings=[],
                top_finding="N/A",
                top_probability=0.0,
                original_image=pil_img,
                heatmap=None,
                overlay=None,
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                device_name=device_str,
                inference_time_ms=elapsed_ms,
                timestamp=timestamp,
                message="Chest X-ray model is not configured.",
                instructions=WEIGHTS_SETUP_INSTRUCTIONS,
                validation_meta=val_res.metadata
            )

        # CASE C: DEMO MODE Enabled (Explicitly Labeled Simulated Run)
        # Execute actual MONAI transforms to exercise the pipeline code
        input_tensor = self.preprocessor.preprocess(pil_img)

        # Determine realistic demo findings based on image intensity distribution
        np_arr = np.array(pil_img.convert("L"))
        h, w = np_arr.shape
        lower_right_zone = np_arr[int(h * 0.5):, int(w * 0.5):]
        density_diff = float(np.mean(lower_right_zone) - np.mean(np_arr))

        if density_diff > 12.0:
            demo_finding = "Pneumonia"
            findings = [
                {"finding": "Pneumonia", "probability": 0.8421, "status": "High"},
                {"finding": "Infiltration", "probability": 0.5640, "status": "High"},
                {"finding": "Consolidation", "probability": 0.4312, "status": "Moderate"},
                {"finding": "Atelectasis", "probability": 0.2810, "status": "Moderate"},
                {"finding": "Pleural Effusion", "probability": 0.1420, "status": "Low"},
                {"finding": "Cardiomegaly", "probability": 0.0890, "status": "Low"},
                {"finding": "Nodule", "probability": 0.0510, "status": "Low"},
                {"finding": "Normal", "probability": 0.0320, "status": "Low"},
            ]
        else:
            demo_finding = "Normal"
            findings = [
                {"finding": "Normal", "probability": 0.9240, "status": "Low"},
                {"finding": "Infiltration", "probability": 0.0980, "status": "Low"},
                {"finding": "Atelectasis", "probability": 0.0760, "status": "Low"},
                {"finding": "Pneumonia", "probability": 0.0520, "status": "Low"},
                {"finding": "Pleural Effusion", "probability": 0.0410, "status": "Low"},
                {"finding": "Cardiomegaly", "probability": 0.0380, "status": "Low"},
                {"finding": "Consolidation", "probability": 0.0210, "status": "Low"},
                {"finding": "Nodule", "probability": 0.0190, "status": "Low"},
            ]

        # Generate simulated anatomical Grad-CAM
        cam_data = generate_demo_gradcam(pil_img, pathology=demo_finding)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return XRayResult(
            status="DEMO_MODE",
            is_demo=True,
            is_configured=False,
            findings=findings,
            top_finding=findings[0]["finding"],
            top_probability=findings[0]["probability"],
            original_image=cam_data["images"]["original"],
            heatmap=cam_data["images"]["heatmap"],
            overlay=cam_data["images"]["overlay"],
            model_name=f"{MODEL_NAME} (Simulated Demo)",
            model_version=MODEL_VERSION,
            device_name=device_str,
            inference_time_ms=elapsed_ms,
            timestamp=timestamp,
            message="DEMO RESULT — NOT A REAL MEDICAL PREDICTION",
            instructions="To switch from Demo Mode to a real model, place weights in models/xray/ and toggle DEMO_MODE to false.",
            validation_meta=val_res.metadata
        )
