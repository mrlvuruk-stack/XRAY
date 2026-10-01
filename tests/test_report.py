"""
Tests for PDF report generation and medical disclaimers.
"""

import sys
from pathlib import Path
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.report import create_medical_pdf, DISCLAIMER_TEXT


def test_pdf_report_generation():
    orig_img = Image.new("RGB", (200, 200), color=(120, 120, 120))
    viz_img = Image.new("RGB", (200, 200), color=(200, 50, 50))

    findings = [
        {"finding": "Pneumonia", "probability": 0.8210, "status": "High"},
        {"finding": "Normal", "probability": 0.0520, "status": "Low"},
    ]

    pdf_bytes = create_medical_pdf(
        modality="Chest X-ray",
        model_name="DenseNet121",
        status="Demo Mode",
        inference_time=0.25,
        findings=findings,
        original_image=orig_img,
        viz_image=viz_img,
        is_demo=True
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF Magic Number verification
    assert pdf_bytes.startswith(b"%PDF-")

