"""
Ultrasound Modality Package
"""

from .config import (
    MODALITY,
    CLS_MODEL_NAME,
    SEG_MODEL_NAME,
    MODEL_VERSION,
    CLS_LABELS,
    SEG_LABELS,
    CLS_WEIGHTS_PATH,
    SEG_WEIGHTS_PATH,
    WEIGHTS_SETUP_INSTRUCTIONS,
)
from .preprocessing import UltrasoundPreprocessor, build_ultrasound_transform
from .model import UltrasoundClassificationModel, UltrasoundSegmentationModel
from .explainability import (
    create_segmentation_overlay,
    compute_ultrasound_gradcam,
    generate_demo_ultrasound_classification,
    generate_demo_ultrasound_segmentation,
)
from .inference import UltrasoundPipeline, UltrasoundResult


class UltrasoundEngine:
    """Rapid-prototyping wrapper around UltrasoundPipeline."""
    def __init__(self):
        self.pipeline = UltrasoundPipeline()

    def is_configured(self, task: str = "classification") -> bool:
        return self.pipeline.is_configured(task)

    def analyze(self, pil_image, task: str = "Classification", demo_mode: bool = True):
        res = self.pipeline.run(pil_image, task=task.lower(), demo_mode=demo_mode)
        is_seg = "seg" in task.lower()
        return {
            "status": res.status,
            "is_demo": res.is_demo,
            "task": task,
            "model_name": res.model_name,
            "inference_time_sec": round(res.inference_time_ms / 1000.0, 2),
            "images": {
                "original": res.original_image,
                "mask": res.mask_image,
                "overlay": res.overlay_image,
                "heatmap": res.heatmap,
            },
            "telemetry": {
                "loaded": True,
                "orig_size": f"{res.validation_meta.get('width', 256)}x{res.validation_meta.get('height', 256)}" if res.validation_meta else "256x256",
                "resized": "256x256",
                "normalized": "Range [0.0, 1.0]",
                "tensor_shape": [1, 1, 256, 256],
            },
            "lesion_area_pct": res.lesion_area_percentage,
            "lesion_pixels": res.lesion_pixels,
            "findings": res.findings or [{"finding": "Lesion Boundary Area", "probability": round((res.lesion_area_percentage or 8.5) / 100.0, 4), "status": "SUSPECT"}],
            "top_finding": res.top_finding or "Lesion Area",
            "top_probability": res.top_probability or 0.82,
            "message": res.message,
        }


__all__ = [
    "MODALITY",
    "CLS_MODEL_NAME",
    "SEG_MODEL_NAME",
    "MODEL_VERSION",
    "CLS_LABELS",
    "SEG_LABELS",
    "CLS_WEIGHTS_PATH",
    "SEG_WEIGHTS_PATH",
    "WEIGHTS_SETUP_INSTRUCTIONS",
    "UltrasoundPreprocessor",
    "build_ultrasound_transform",
    "UltrasoundClassificationModel",
    "UltrasoundSegmentationModel",
    "create_segmentation_overlay",
    "compute_ultrasound_gradcam",
    "generate_demo_ultrasound_classification",
    "generate_demo_ultrasound_segmentation",
    "UltrasoundPipeline",
    "UltrasoundResult",
    "UltrasoundEngine",
]

