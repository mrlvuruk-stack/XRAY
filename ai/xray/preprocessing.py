"""
Chest X-Ray Preprocessing Pipeline
Utilizes MONAI medical imaging transforms for standardizing radiographs.
"""

from typing import Union, Tuple
import numpy as np
from PIL import Image
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Resize,
    ScaleIntensityRange,
    EnsureType,
)

try:
    from .config import INPUT_SIZE, IN_CHANNELS
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ai.xray.config import INPUT_SIZE, IN_CHANNELS



def build_xray_transform(spatial_size: Tuple[int, int] = INPUT_SIZE) -> Compose:
    """
    Builds the core MONAI transform pipeline for Chest X-Ray images.
    1. Channel formatting: Ensures (1, H, W)
    2. Resizing: Standardizes spatial dimensions to input size
    3. Intensity scaling: Maps pixel intensities [0, 255] -> [0.0, 1.0]
    4. Tensor conversion: Ensures float32 PyTorch tensor
    """
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        Resize(spatial_size=spatial_size, mode="bilinear", anti_aliasing=True),
        ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        EnsureType(data_type="tensor", dtype=torch.float32),
    ])


class XRayPreprocessor:
    """
    Preprocessor handling input conversion and MONAI transformation.
    """

    def __init__(self, spatial_size: Tuple[int, int] = INPUT_SIZE):
        self.spatial_size = spatial_size
        self.transform = build_xray_transform(spatial_size=spatial_size)

    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Converts PIL Image or numpy array to a standardized 4D Tensor (1, 1, H, W).
        """
        if isinstance(image, Image.Image):
            # Convert to single-channel grayscale for chest radiography
            grayscale_img = image.convert("L")
            np_data = np.array(grayscale_img, dtype=np.float32)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                # Average channels if RGB
                np_data = np.mean(image, axis=-1).astype(np.float32)
            else:
                np_data = image.astype(np.float32)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Execute MONAI transforms: returns (1, H, W) MetaTensor
        tensor_data = self.transform(np_data)

        # Add batch dimension: (1, 1, H, W)
        if tensor_data.ndim == 3:
            tensor_data = tensor_data.unsqueeze(0)

        return tensor_data


if __name__ == "__main__":
    preprocessor = XRayPreprocessor()
    test_img = np.zeros((300, 300), dtype=np.float32)
    output = preprocessor.preprocess(test_img)
    print(f"X-Ray preprocessing pipeline operational! Processed tensor shape: {output.shape}")

