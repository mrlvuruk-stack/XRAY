"""
Tests for Ultrasound dual pipeline: classification and segmentation.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.ultrasound.model import UltrasoundClassificationModel, UltrasoundSegmentationModel
from ai.ultrasound.inference import UltrasoundPipeline, UltrasoundResult


def create_sample_ultrasound():
    arr = np.random.uniform(20, 180, (256, 256)).astype(np.uint8)
    return Image.fromarray(arr)


def test_ultrasound_models_unconfigured():
    cls_m = UltrasoundClassificationModel(weights_path="fake_cls.pth")
    seg_m = UltrasoundSegmentationModel(weights_path="fake_seg.pth")
    assert cls_m.is_configured() is False
    assert seg_m.is_configured() is False


def test_ultrasound_pipeline_unconfigured():
    pipeline = UltrasoundPipeline(cls_weights_path="fake_cls.pth", seg_weights_path="fake_seg.pth")
    img = create_sample_ultrasound()

    # Classification
    res_cls = pipeline.run(img, task="classification", demo_mode=False)
    assert res_cls.status == "MODEL_NOT_CONFIGURED"
    assert "Ultrasound classification model is not configured." in res_cls.message

    # Segmentation
    res_seg = pipeline.run(img, task="segmentation", demo_mode=False)
    assert res_seg.status == "MODEL_NOT_CONFIGURED"
    assert "Ultrasound segmentation model is not configured." in res_seg.message


def test_ultrasound_pipeline_demo_classification():
    pipeline = UltrasoundPipeline(cls_weights_path="fake_cls.pth", seg_weights_path="fake_seg.pth")
    img = create_sample_ultrasound()

    res = pipeline.run(img, task="classification", demo_mode=True)
    assert res.status == "DEMO_MODE"
    assert res.is_demo is True
    assert "DEMO RESULT — NOT A REAL MEDICAL PREDICTION" in res.message
    assert len(res.findings) == 3
    assert res.heatmap is not None


def test_ultrasound_pipeline_demo_segmentation():
    pipeline = UltrasoundPipeline(cls_weights_path="fake_cls.pth", seg_weights_path="fake_seg.pth")
    img = create_sample_ultrasound()

    res = pipeline.run(img, task="segmentation", demo_mode=True)
    assert res.status == "DEMO_MODE"
    assert res.is_demo is True
    assert "DEMO RESULT — NOT A REAL MEDICAL PREDICTION" in res.message
    assert res.mask_image is not None
    assert res.overlay_image is not None
    assert res.lesion_area_percentage is not None
