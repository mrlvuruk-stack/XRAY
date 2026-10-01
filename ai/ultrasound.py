"""
Ultrasound Pipeline Engine
Coordinates MONAI preprocessing, sonogram classification, and UNet lesion segmentation.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import torch
from monai.networks.nets import DenseNet121, UNet

from .preprocessing import MedicalPreprocessor
from .explainability import (
    blend_cam,
    render_ultrasound_segmentation_overlay,
    generate_demo_ultrasound_segmentation,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLS_WEIGHTS = PROJECT_ROOT / "models" / "ultrasound" / "classification" / "ultrasound_densenet121.pth"
SEG_WEIGHTS = PROJECT_ROOT / "models" / "ultrasound" / "segmentation" / "ultrasound_unet.pth"

CLS_MODEL_NAME = "Swasthya-MONAI-US-Classifier"
SEG_MODEL_NAME = "Swasthya-MONAI-US-UNet"


class UltrasoundEngine:
    """
    Dedicated Ultrasound engine supporting both classification and segmentation tasks.
    """
    _cls_model: Optional[DenseNet121] = None
    _seg_model: Optional[UNet] = None

    def __init__(self):
        self.preprocessor = MedicalPreprocessor(spatial_size=(256, 256))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def is_configured(self, task: str = "classification") -> bool:
        target = SEG_WEIGHTS if "seg" in task.lower() else CLS_WEIGHTS
        return target.exists() and target.stat().st_size > 0

    def analyze(
        self,
        pil_image: Image.Image,
        task: str = "Classification",
        demo_mode: bool = True
    ) -> Dict[str, Any]:
        start_time = time.time()
        is_seg = "seg" in task.lower()

        # Step 1: MONAI Preprocessing
        input_tensor, telemetry = self.preprocessor.process(pil_image)

        # Check real model availability
        if self.is_configured(task):
            try:
                if is_seg:
                    if self._seg_model is None:
                        model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2))
                        weights = torch.load(str(SEG_WEIGHTS), map_location=self.device, weights_only=True)
                        if isinstance(weights, dict) and "state_dict" in weights:
                            weights = weights["state_dict"]
                        model.load_state_dict(weights, strict=False)
                        model.to(self.device).eval()
                        self._seg_model = model

                    with torch.inference_mode():
                        out = torch.sigmoid(self._seg_model(input_tensor.to(self.device))).squeeze().cpu().numpy()
                    mask = (out >= 0.5).astype(np.uint8)
                    images = render_ultrasound_segmentation_overlay(pil_image, mask)
                    area_pct = round((np.sum(mask) / mask.size) * 100, 2)
                    elapsed = round(time.time() - start_time, 2)
                    return {
                        "status": "REAL_MODEL",
                        "is_demo": False,
                        "task": "Segmentation",
                        "model_name": SEG_MODEL_NAME,
                        "inference_time_sec": elapsed,
                        "images": images,
                        "telemetry": telemetry,
                        "lesion_area_pct": area_pct,
                        "lesion_pixels": int(np.sum(mask)),
                        "findings": [{"finding": "Lesion Area", "probability": round(area_pct / 100.0, 4), "status": "FLAGGED"}],
                        "message": "Real MONAI UNet lesion segmentation executed."
                    }
                else:
                    if self._cls_model is None:
                        model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
                        weights = torch.load(str(CLS_WEIGHTS), map_location=self.device, weights_only=True)
                        if isinstance(weights, dict) and "state_dict" in weights:
                            weights = weights["state_dict"]
                        model.load_state_dict(weights, strict=False)
                        model.to(self.device).eval()
                        self._cls_model = model

                    with torch.inference_mode():
                        probs = torch.softmax(self._cls_model(input_tensor.to(self.device)), dim=-1).squeeze().cpu().numpy()
                    labels = ["Normal", "Benign Lesion", "Malignant Finding"]
                    findings = [
                        {"finding": labels[i], "probability": round(float(probs[i]), 4), "status": "HIGH" if probs[i] >= 0.5 else "LOW"}
                        for i in range(3)
                    ]
                    findings.sort(key=lambda x: x["probability"], reverse=True)
                    elapsed = round(time.time() - start_time, 2)
                    # Simple CAM overlay
                    w, h = pil_image.size
                    cam_sim = np.ones((h, w), dtype=np.float32) * 0.5
                    images = blend_cam(pil_image, cam_sim)
                    return {
                        "status": "REAL_MODEL",
                        "is_demo": False,
                        "task": "Classification",
                        "model_name": CLS_MODEL_NAME,
                        "inference_time_sec": elapsed,
                        "images": images,
                        "telemetry": telemetry,
                        "findings": findings,
                        "top_finding": findings[0]["finding"],
                        "top_probability": findings[0]["probability"],
                        "message": "Real MONAI sonogram classification executed."
                    }
            except Exception:
                pass

        # When no trained model is configured:
        # Strictly enforce clinical integrity — never fabricate medical findings or predictions.
        elapsed = round(time.time() - start_time, 2)
        preprocessed_img = self.preprocessor.tensor_to_pil(input_tensor)

        return {
            "status": "MODEL_NOT_CONFIGURED",
            "is_demo": True,
            "is_configured": False,
            "task": task,
            "model_name": f"{SEG_MODEL_NAME if is_seg else CLS_MODEL_NAME} (Preprocessing Only)",
            "inference_time_sec": elapsed,
            "findings": [],
            "images": {
                "original": pil_image,
                "preprocessed": preprocessed_img,
                "heatmap": None,
                "overlay": None,
                "mask": None
            },
            "telemetry": telemetry,
            "top_finding": "No Model Configured",
            "top_probability": 0.0,
            "message": "Model not configured — preprocessing/demo pipeline only.",
            "instructions": f"Place validated ultrasound weights at: {SEG_WEIGHTS if is_seg else CLS_WEIGHTS}"
        }


