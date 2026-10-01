"""
Image Validation Module
Validates medical image files, checks integrity, prevents directory traversal,
and ensures dimensions/channels meet diagnostic pipeline requirements.
"""

import io
import re
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageOps

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".dcm", ".dicom"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MIN_RESOLUTION = (32, 32)
MAX_RESOLUTION = (8192, 8192)


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    sanitized_filename: str = "image.png"
    metadata: Dict[str, Any] = None
    pil_image: Optional[Image.Image] = None


def sanitize_filename(filename: str) -> str:
    """
    Strips directory traversal sequences and removes unsafe characters.
    Never exposes internal filesystem paths.
    """
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean_name)
    if not clean_name:
        clean_name = "uploaded_scan.png"
    return clean_name


def validate_medical_image(
    file_bytes_or_path: Any,
    filename: str = "scan.png",
    modality: str = "generic"
) -> ValidationResult:
    """
    Validates uploaded scan bytes or path against medical imaging constraints.
    Prevents path leakage, checks for corruption, dimension validity, and intensity variation.
    """
    sanitized_name = sanitize_filename(filename)
    ext = os.path.splitext(sanitized_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return ValidationResult(
            is_valid=False,
            error_message=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            sanitized_filename=sanitized_name
        )

    # Handle direct PIL Image or Numpy array
    if isinstance(file_bytes_or_path, Image.Image):
        img = file_bytes_or_path
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        raw_bytes = bio.getvalue()
    elif isinstance(file_bytes_or_path, np.ndarray):
        img = Image.fromarray(file_bytes_or_path.astype(np.uint8))
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        raw_bytes = bio.getvalue()
    # Load bytes from file-like or path
    elif isinstance(file_bytes_or_path, (bytes, bytearray)):
        raw_bytes = bytes(file_bytes_or_path)
    elif hasattr(file_bytes_or_path, "read"):
        raw_bytes = file_bytes_or_path.read()
        if hasattr(file_bytes_or_path, "seek"):
            file_bytes_or_path.seek(0)
    elif isinstance(file_bytes_or_path, str) and os.path.exists(file_bytes_or_path):
        try:
            with open(file_bytes_or_path, "rb") as f:
                raw_bytes = f.read()
        except Exception:
            return ValidationResult(
                is_valid=False,
                error_message="Unable to read image data from specified path.",
                sanitized_filename=sanitized_name
            )
    else:
        return ValidationResult(
            is_valid=False,
            error_message="Invalid file data provided.",
            sanitized_filename=sanitized_name
        )


    # Size check
    if len(raw_bytes) == 0:
        return ValidationResult(
            is_valid=False,
            error_message="Uploaded file is empty (0 bytes).",
            sanitized_filename=sanitized_name
        )

    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        return ValidationResult(
            is_valid=False,
            error_message=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
            sanitized_filename=sanitized_name
        )

    # Decode check
    try:
        bio = io.BytesIO(raw_bytes)
        img = Image.open(bio)
        img.verify()  # Verify integrity
        
        # Re-open after verify (PIL requirement)
        bio.seek(0)
        img = Image.open(bio)
        img.load()
    except Exception:
        return ValidationResult(
            is_valid=False,
            error_message="Corrupted or unreadable image file. Please provide a valid diagnostic scan.",
            sanitized_filename=sanitized_name
        )

    width, height = img.size
    if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
        return ValidationResult(
            is_valid=False,
            error_message=f"Image resolution {width}x{height} is too low for diagnostic processing (min {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]}).",
            sanitized_filename=sanitized_name
        )

    if width > MAX_RESOLUTION[0] or height > MAX_RESOLUTION[1]:
        return ValidationResult(
            is_valid=False,
            error_message=f"Image resolution {width}x{height} exceeds maximum threshold of {MAX_RESOLUTION[0]}x{MAX_RESOLUTION[1]}.",
            sanitized_filename=sanitized_name
        )

    # Variance / Blank image check
    np_arr = np.array(img.convert("L"))
    variance = float(np.var(np_arr))
    if variance < 1e-4:
        return ValidationResult(
            is_valid=False,
            error_message="Image contains uniform pixel intensity (blank or solid color). Please upload a valid scan.",
            sanitized_filename=sanitized_name
        )

    # Standardize image orientation if EXIF present
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    metadata = {
        "width": width,
        "height": height,
        "mode": img.mode,
        "format": ext.replace(".", "").upper(),
        "size_kb": round(len(raw_bytes) / 1024, 2),
        "mean_intensity": round(float(np.mean(np_arr)), 2),
        "std_intensity": round(float(np.std(np_arr)), 2),
        "modality": modality,
    }

    return ValidationResult(
        is_valid=True,
        error_message=None,
        sanitized_filename=sanitized_name,
        metadata=metadata,
        pil_image=img
    )
