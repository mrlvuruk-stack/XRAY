"""
Chest X-Ray Modality Package
"""

from .config import (
    MODEL_NAME,
    MODEL_VERSION,
    MODALITY,
    TASK,
    LABELS,
    DEFAULT_WEIGHTS_PATH,
    WEIGHTS_SETUP_INSTRUCTIONS,
)
from .preprocessing import XRayPreprocessor, build_xray_transform
from .model import XRayDenseNetModel
from .explainability import compute_xray_gradcam, blend_heatmap_with_image
from .inference import XRayPipeline, XRayResult


class XRayEngine:
    """Rapid-prototyping wrapper around XRayPipeline."""
    def __init__(self):
        self.pipeline = XRayPipeline()

    def is_configured(self) -> bool:
        return self.pipeline.is_model_configured()

    def analyze(self, pil_image, demo_mode: bool = True):
        res = self.pipeline.run(pil_image, demo_mode=demo_mode)
        return {
            "status": res.status,
            "is_demo": res.is_demo,
            "model_name": res.model_name,
            "inference_time_sec": round(res.inference_time_ms / 1000.0, 2),
            "findings": res.findings,
            "images": {
                "original": res.original_image,
                "heatmap": res.heatmap,
                "overlay": res.overlay,
            },
            "telemetry": {
                "loaded": True,
                "orig_size": f"{res.validation_meta.get('width', 224)}x{res.validation_meta.get('height', 224)}" if res.validation_meta else "224x224",
                "resized": "224x224",
                "normalized": "Range [0.0, 1.0]",
                "tensor_shape": [1, 1, 224, 224],
            },
            "top_finding": res.top_finding,
            "top_probability": res.top_probability,
            "message": res.message,
            "instructions": res.instructions,
        }


__all__ = [
    "MODEL_NAME",
    "MODEL_VERSION",
    "MODALITY",
    "TASK",
    "LABELS",
    "DEFAULT_WEIGHTS_PATH",
    "WEIGHTS_SETUP_INSTRUCTIONS",
    "XRayPreprocessor",
    "build_xray_transform",
    "XRayDenseNetModel",
    "compute_xray_gradcam",
    "blend_heatmap_with_image",
    "XRayPipeline",
    "XRayResult",
    "XRayEngine",
]

