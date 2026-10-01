"""
Ultrasound Preprocessing Module
Applies MONAI transforms tailored to ultrasound B-mode images.
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
    from .config import INPUT_SIZE
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ai.ultrasound.config import INPUT_SIZE


def build_ultrasound_transform(spatial_size: Tuple[int, int] = INPUT_SIZE) -> Compose:
    """
    Builds the MONAI transform pipeline for Ultrasound B-mode images.
    1. Single channel formatting (1, H, W)
    2. Resizing to standard ultrasound field of view (256, 256)
    3. Normalization to [0.0, 1.0] range
    4. PyTorch float32 tensor conversion
    """
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        Resize(spatial_size=spatial_size, mode="bilinear", anti_aliasing=True),
        ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        EnsureType(data_type="tensor", dtype=torch.float32),
    ])


class UltrasoundPreprocessor:
    """
    Preprocessor for converting sonogram images using MONAI transforms.
    """

    def __init__(self, spatial_size: Tuple[int, int] = INPUT_SIZE):
        self.spatial_size = spatial_size
        self.transform = build_ultrasound_transform(spatial_size=spatial_size)

    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Converts input to standardized 4D Tensor (1, 1, H, W).
        """
        if isinstance(image, Image.Image):
            grayscale_img = image.convert("L")
            np_data = np.array(grayscale_img, dtype=np.float32)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                np_data = np.mean(image, axis=-1).astype(np.float32)
            else:
                np_data = image.astype(np.float32)
        else:
            raise ValueError(f"Unsupported ultrasound image type: {type(image)}")

        tensor_data = self.transform(np_data)

        if tensor_data.ndim == 3:
            tensor_data = tensor_data.unsqueeze(0)

        return tensor_data


if __name__ == "__main__":
    preprocessor = UltrasoundPreprocessor()
    test_img = np.zeros((320, 320), dtype=np.float32)
    output = preprocessor.preprocess(test_img)
    print(f"Ultrasound preprocessing operational! Output tensor shape: {output.shape}")
