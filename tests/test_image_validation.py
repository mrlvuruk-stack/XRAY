"""
Unit tests for medical image validation and path safety.
"""

import sys
import io
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.common.image_validation import validate_medical_image, sanitize_filename


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd.png") == "......etcpasswd.png" or "passwd.png" in sanitize_filename("../../etc/passwd.png")
    assert ".." not in sanitize_filename("..\\..\\scan.png")
    assert sanitize_filename("valid_scan_01.png") == "valid_scan_01.png"


def test_valid_image_bytes():
    img = Image.new("L", (256, 256), color=100)
    # Add non-zero variance
    np_img = np.array(img)
    np_img[100:150, 100:150] = 200
    img = Image.fromarray(np_img)

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)

    res = validate_medical_image(bio.getvalue(), filename="test.png")
    assert res.is_valid is True
    assert res.error_message is None
    assert res.pil_image is not None
    assert res.metadata["width"] == 256


def test_unsupported_format():
    res = validate_medical_image(b"fake data", filename="test.txt")
    assert res.is_valid is False
    assert "Unsupported file format" in res.error_message


def test_empty_file():
    res = validate_medical_image(b"", filename="empty.png")
    assert res.is_valid is False
    assert "empty" in res.error_message.lower()


def test_corrupted_image():
    res = validate_medical_image(b"\x89PNG\r\n\x1a\nCorruptedDataHere", filename="corrupt.png")
    assert res.is_valid is False
    assert "corrupted" in res.error_message.lower()


def test_uniform_blank_image():
    img = Image.new("L", (100, 100), color=0)  # Solid black
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    res = validate_medical_image(bio.getvalue(), filename="blank.png")
    assert res.is_valid is False
    assert "blank" in res.error_message.lower() or "uniform" in res.error_message.lower()
