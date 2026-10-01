"""
Tests for Chest X-Ray pipeline, model unconfigured state, demo mode, and Grad-CAM.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.xray.model import XRayDenseNetModel
from ai.xray.inference import XRayPipeline, XRayResult


def create_sample_xray():
    arr = np.random.uniform(50, 200, (256, 256)).astype(np.uint8)
    return Image.fromarray(arr)


def test_xray_model_unconfigured():
    model = XRayDenseNetModel(weights_path="non_existent_weights.pth")
    assert model.is_configured() is False
    assert model.load_weights() is False


def test_xray_pipeline_unconfigured_when_demo_false():
    pipeline = XRayPipeline(weights_path="non_existent_weights.pth")
    img = create_sample_xray()

    result = pipeline.run(image_input=img, demo_mode=False)
    assert result.status == "MODEL_NOT_CONFIGURED"
    assert result.is_configured is False
    assert result.is_demo is False
    assert "Chest X-ray model is not configured." in result.message
    assert result.findings == []


def test_xray_pipeline_demo_mode():
    pipeline = XRayPipeline(weights_path="non_existent_weights.pth")
    img = create_sample_xray()

    result = pipeline.run(image_input=img, demo_mode=True)
    assert result.status == "DEMO_MODE"
    assert result.is_demo is True
    assert result.is_configured is False
    assert "DEMO RESULT — NOT A REAL MEDICAL PREDICTION" in result.message
    assert len(result.findings) > 0
    assert result.heatmap is not None
    assert result.overlay is not None
    assert result.original_image is not None
