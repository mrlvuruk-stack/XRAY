"""
MONAI Preprocessing Engine
Implements standardized medical image transforms using MONAI.
"""

from typing import Tuple, Dict, Any, Union
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


def get_monai_transforms(spatial_size: Tuple[int, int] = (224, 224)) -> Compose:
    """
    Constructs the core MONAI transform pipeline.
    """
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        Resize(spatial_size=spatial_size, mode="bilinear", anti_aliasing=True),
        ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        EnsureType(data_type="tensor", dtype=torch.float32),
    ])


class MedicalPreprocessor:
    """
    Wraps MONAI transforms and returns both the tensor and the execution verification telemetry.
    """

    def __init__(self, spatial_size: Tuple[int, int] = (224, 224)):
        self.spatial_size = spatial_size
        self.transform = get_monai_transforms(spatial_size)

    def process(self, image: Union[Image.Image, np.ndarray]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes MONAI preprocessing and generates verified step-by-step trace.
        """
        # Step 1: Image loading & array conversion
        if isinstance(image, Image.Image):
            gray_img = image.convert("L")
            np_data = np.array(gray_img, dtype=np.float32)
            orig_size = image.size
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                np_data = np.mean(image, axis=-1).astype(np.float32)
            else:
                np_data = image.astype(np.float32)
            orig_size = (np_data.shape[1], np_data.shape[0])
        else:
            raise ValueError(f"Unsupported image input type: {type(image)}")

        # Step 2 & 3: MONAI transforms (EnsureChannelFirst, Resize, ScaleIntensityRange, EnsureType)
        tensor_data = self.transform(np_data)

        # Step 4: Batch dimension (1, 1, H, W)
        if tensor_data.ndim == 3:
            tensor_data = tensor_data.unsqueeze(0)

        telemetry = {
            "loaded": True,
            "orig_size": f"{orig_size[0]}x{orig_size[1]}",
            "resized": f"{self.spatial_size[0]}x{self.spatial_size[1]}",
            "normalized": "Range [0.0, 1.0]",
            "tensor_shape": list(tensor_data.shape),
            "dtype": str(tensor_data.dtype).replace("torch.", ""),
        }

        return tensor_data, telemetry

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Converts normalized MONAI float32 tensor back to a displayable PIL image."""
        arr = tensor.squeeze().detach().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

