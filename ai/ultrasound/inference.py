"""
Ultrasound Inference Pipeline
Supports dual-mode processing:
1. Classification (Normal vs Benign vs Malignant)
2. Segmentation (MONAI UNet Lesion Masking)
Enforces REAL MODEL vs DEMO MODE vs MODEL NOT CONFIGURED states.
"""

import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from PIL import Image
import torch

from ..common.device import get_device, get_device_info
from ..common.image_validation import validate_medical_image, ValidationResult
from .config import (
    CLS_MODEL_NAME,
    SEG_MODEL_NAME,
    MODEL_VERSION,
    CLS_WEIGHTS_PATH,
    SEG_WEIGHTS_PATH,
    WEIGHTS_SETUP_INSTRUCTIONS,
)
from .preprocessing import UltrasoundPreprocessor
from .model import UltrasoundClassificationModel, UltrasoundSegmentationModel
from .explainability import (
    generate_demo_ultrasound_classification,
    generate_demo_ultrasound_segmentation,
)


@dataclass
class UltrasoundResult:
    """Standardized result for Ultrasound analysis."""
    modality: str  # "ultrasound"
    task: str      # "classification" | "segmentation"
    status: str    # "REAL_MODEL" | "DEMO_MODE" | "MODEL_NOT_CONFIGURED" | "ERROR"
    is_demo: bool
    is_configured: bool
    model_name: str
    model_version: str
    device_name: str
    inference_time_ms: float
    timestamp: str
    original_image: Optional[Image.Image]
    # Classification fields
    findings: Optional[List[Dict[str, Any]]] = None
    top_finding: Optional[str] = None
    top_probability: Optional[float] = None
    heatmap: Optional[Image.Image] = None
    # Segmentation fields
    mask_image: Optional[Image.Image] = None
    overlay_image: Optional[Image.Image] = None
    lesion_pixels: Optional[int] = None
    lesion_area_percentage: Optional[float] = None
    # Status / instructions
    message: Optional[str] = None
    instructions: Optional[str] = None
    validation_meta: Optional[Dict[str, Any]] = None


class UltrasoundPipeline:
    """
    Dedicated Ultrasound diagnostic pipeline supporting both classification and segmentation.
    """

    def __init__(
        self,
        cls_weights_path: Optional[str] = None,
        seg_weights_path: Optional[str] = None
    ):
        self.device = get_device()
        self.preprocessor = UltrasoundPreprocessor()
        self.cls_model = UltrasoundClassificationModel(weights_path=cls_weights_path or str(CLS_WEIGHTS_PATH))
        self.seg_model = UltrasoundSegmentationModel(weights_path=seg_weights_path or str(SEG_WEIGHTS_PATH))

    def is_configured(self, task: str = "classification") -> bool:
        if task == "segmentation":
            return self.seg_model.is_configured()
        return self.cls_model.is_configured()

    def run(
        self,
        image_input: Any,
        task: str = "classification",
        filename: str = "ultrasound_scan.png",
        demo_mode: bool = False
    ) -> UltrasoundResult:
        task = task.lower()
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dev_info = get_device_info()
        dev_str = dev_info["device_str"]

        model_name = SEG_MODEL_NAME if task == "segmentation" else CLS_MODEL_NAME

        # Step 1: Image Validation
        val_res = validate_medical_image(
            file_bytes_or_path=image_input,
            filename=filename,
            modality="ultrasound"
        )
        if not val_res.is_valid:
            return UltrasoundResult(
                modality="ultrasound",
                task=task,
                status="ERROR",
                is_demo=demo_mode,
                is_configured=self.is_configured(task),
                model_name=model_name,
                model_version=MODEL_VERSION,
                device_name=dev_str,
                inference_time_ms=0.0,
                timestamp=timestamp,
                original_image=None,
                message=val_res.error_message,
                validation_meta=None
            )

        pil_img = val_res.pil_image
        is_configured = self.is_configured(task)

        # CASE A: Real Model Configured with Weights
        if is_configured:
            try:
                tensor = self.preprocessor.preprocess(pil_img)
                if task == "segmentation":
                    if not self.seg_model._is_loaded:
                        self.seg_model.load_weights(self.device)
                    seg_pred = self.seg_model.predict(tensor)
                    overlay_data = self.seg_model.explain(pil_img, tensor)
                    elapsed_ms = round((time.time() - start_time) * 1000, 2)
                    return UltrasoundResult(
                        modality="ultrasound",
                        task="segmentation",
                        status="REAL_MODEL",
                        is_demo=False,
                        is_configured=True,
                        model_name=model_name,
                        model_version=MODEL_VERSION,
                        device_name=dev_str,
                        inference_time_ms=elapsed_ms,
                        timestamp=timestamp,
                        original_image=overlay_data["original"],
                        mask_image=overlay_data["mask"],
                        overlay_image=overlay_data["overlay"],
                        lesion_pixels=seg_pred["lesion_pixels"],
                        lesion_area_percentage=seg_pred["lesion_area_percentage"],
                        message="Real UNet lesion segmentation executed.",
                        validation_meta=val_res.metadata
                    )
                else:
                    if not self.cls_model._is_loaded:
                        self.cls_model.load_weights(self.device)
                    cls_pred = self.cls_model.predict(tensor)
                    top_idx = 0
                    for i, l in enumerate(self.cls_model.labels):
                        if l == cls_pred["top_finding"]:
                            top_idx = i
                            break
                    cam_data = self.cls_model.explain(pil_img, tensor, target_class=top_idx)
                    elapsed_ms = round((time.time() - start_time) * 1000, 2)
                    return UltrasoundResult(
                        modality="ultrasound",
                        task="classification",
                        status="REAL_MODEL",
                        is_demo=False,
                        is_configured=True,
                        model_name=model_name,
                        model_version=MODEL_VERSION,
                        device_name=dev_str,
                        inference_time_ms=elapsed_ms,
                        timestamp=timestamp,
                        original_image=cam_data["images"]["original"],
                        heatmap=cam_data["images"]["heatmap"],
                        overlay_image=cam_data["images"]["overlay"],
                        findings=cls_pred["findings"],
                        top_finding=cls_pred["top_finding"],
                        top_probability=cls_pred["top_probability"],
                        message="Real DenseNet sonogram classification executed.",
                        validation_meta=val_res.metadata
                    )
            except Exception as e:
                elapsed_ms = round((time.time() - start_time) * 1000, 2)
                return UltrasoundResult(
                    modality="ultrasound",
                    task=task,
                    status="ERROR",
                    is_demo=False,
                    is_configured=True,
                    model_name=model_name,
                    model_version=MODEL_VERSION,
                    device_name=dev_str,
                    inference_time_ms=elapsed_ms,
                    timestamp=timestamp,
                    original_image=pil_img,
                    message="An error occurred during ultrasound model execution.",
                    validation_meta=val_res.metadata
                )

        # CASE B: Model Weights Not Configured, Demo Mode Disabled
        if not demo_mode:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            setup_text = WEIGHTS_SETUP_INSTRUCTIONS.get(task, "")
            return UltrasoundResult(
                modality="ultrasound",
                task=task,
                status="MODEL_NOT_CONFIGURED",
                is_demo=False,
                is_configured=False,
                model_name=model_name,
                model_version=MODEL_VERSION,
                device_name=dev_str,
                inference_time_ms=elapsed_ms,
                timestamp=timestamp,
                original_image=pil_img,
                message=f"Ultrasound {task} model is not configured.",
                instructions=setup_text,
                validation_meta=val_res.metadata
            )

        # CASE C: DEMO MODE Enabled
        # Execute MONAI preprocessor to exercise medical transform logic
        tensor = self.preprocessor.preprocess(pil_img)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        demo_msg = "DEMO RESULT — NOT A REAL MEDICAL PREDICTION"

        if task == "segmentation":
            demo_seg = generate_demo_ultrasound_segmentation(pil_img)
            return UltrasoundResult(
                modality="ultrasound",
                task="segmentation",
                status="DEMO_MODE",
                is_demo=True,
                is_configured=False,
                model_name=f"{SEG_MODEL_NAME} (Simulated Demo)",
                model_version=MODEL_VERSION,
                device_name=dev_str,
                inference_time_ms=elapsed_ms,
                timestamp=timestamp,
                original_image=demo_seg["images"]["original"],
                mask_image=demo_seg["images"]["mask"],
                overlay_image=demo_seg["images"]["overlay"],
                lesion_pixels=demo_seg["lesion_pixels"],
                lesion_area_percentage=demo_seg["lesion_area_percentage"],
                message=demo_msg,
                instructions="To use a real segmentation model, download UNet weights to models/ultrasound/segmentation/.",
                validation_meta=val_res.metadata
            )
        else:
            demo_cam = generate_demo_ultrasound_classification(pil_img)
            findings = [
                {"finding": "Benign Lesion", "probability": 0.8240, "status": "High"},
                {"finding": "Normal", "probability": 0.1250, "status": "Low"},
                {"finding": "Malignant Finding", "probability": 0.0510, "status": "Low"},
            ]
            return UltrasoundResult(
                modality="ultrasound",
                task="classification",
                status="DEMO_MODE",
                is_demo=True,
                is_configured=False,
                model_name=f"{CLS_MODEL_NAME} (Simulated Demo)",
                model_version=MODEL_VERSION,
                device_name=dev_str,
                inference_time_ms=elapsed_ms,
                timestamp=timestamp,
                original_image=demo_cam["images"]["original"],
                heatmap=demo_cam["images"]["heatmap"],
                overlay_image=demo_cam["images"]["overlay"],
                findings=findings,
                top_finding=findings[0]["finding"],
                top_probability=findings[0]["probability"],
                message=demo_msg,
                instructions="To use a real classification model, download DenseNet weights to models/ultrasound/classification/.",
                validation_meta=val_res.metadata
            )
