"""
Unit tests for MONAI preprocessing pipelines in X-Ray and Ultrasound.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.xray.preprocessing import XRayPreprocessor
from ai.ultrasound.preprocessing import UltrasoundPreprocessor


def test_xray_preprocessing_pil():
    preprocessor = XRayPreprocessor(spatial_size=(224, 224))
    pil_img = Image.new("L", (350, 400), color=128)
    tensor = preprocessor.preprocess(pil_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 4
    assert tensor.shape == (1, 1, 224, 224)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min().item() <= tensor.max().item() <= 1.0


def test_xray_preprocessing_numpy():
    preprocessor = XRayPreprocessor(spatial_size=(224, 224))
    np_img = np.random.uniform(0, 255, (300, 300)).astype(np.float32)
    tensor = preprocessor.preprocess(np_img)

    assert tensor.shape == (1, 1, 224, 224)
    assert tensor.dtype == torch.float32


def test_ultrasound_preprocessing_pil():
    preprocessor = UltrasoundPreprocessor(spatial_size=(256, 256))
    pil_img = Image.new("RGB", (320, 320), color=(100, 100, 100))
    tensor = preprocessor.preprocess(pil_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 4
    assert tensor.shape == (1, 1, 256, 256)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min().item() <= tensor.max().item() <= 1.0


def test_ultrasound_preprocessing_numpy():
    preprocessor = UltrasoundPreprocessor(spatial_size=(256, 256))
    np_img = np.random.uniform(0, 255, (256, 256, 3)).astype(np.float32)
    tensor = preprocessor.preprocess(np_img)

    assert tensor.shape == (1, 1, 256, 256)
    assert tensor.dtype == torch.float32
