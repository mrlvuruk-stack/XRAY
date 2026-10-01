"""
Chest X-Ray Explainability Module
Generates Grad-CAM visual heatmaps and overlays using MONAI GradCAM utilities.
"""

from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
import cv2
import matplotlib.cm as cm
from monai.visualize import GradCAM


def blend_heatmap_with_image(
    pil_image: Image.Image,
    cam_array: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet"
) -> Dict[str, Image.Image]:
    """
    Blends a 2D activation array [0, 1] with the original radiograph.
    Returns Original, Heatmap, and Overlay as PIL Images.
    """
    orig_rgb = pil_image.convert("RGB")
    width, height = orig_rgb.size

    # Ensure cam_array is 2D and normalized
    cam_2d = np.squeeze(cam_array)
    cam_min, cam_max = np.min(cam_2d), np.max(cam_2d)
    if cam_max > cam_min:
        cam_norm = (cam_2d - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.zeros_like(cam_2d)

    # Resize CAM to match original image resolution
    cam_resized = cv2.resize(cam_norm, (width, height), interpolation=cv2.INTER_LINEAR)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    # Apply colormap using modern Matplotlib API
    import matplotlib as mpl
    cmap = mpl.colormaps.get(colormap_name, mpl.colormaps["jet"])
    heatmap_colored = cmap(cam_resized)[:, :, :3]  # Drop alpha, float [0, 1]

    heatmap_uint8 = np.uint8(255 * heatmap_colored)
    heatmap_pil = Image.fromarray(heatmap_uint8)

    # Blend with original
    orig_np = np.array(orig_rgb, dtype=np.float32)
    overlay_np = (alpha * heatmap_uint8 + (1.0 - alpha) * orig_np).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_np)

    return {
        "original": orig_rgb,
        "heatmap": heatmap_pil,
        "overlay": overlay_pil,
    }


def compute_xray_gradcam(
    model: torch.nn.Module,
    target_layer: str,
    input_tensor: torch.Tensor,
    pil_image: Image.Image,
    target_class: Optional[int] = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Computes real Grad-CAM activations using MONAI GradCAM.
    """
    if device is None:
        device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)
    target_class_idx = target_class if target_class is not None else 0

    try:
        gradcam = GradCAM(
            nn_module=model,
            target_layers=target_layer,
        )
        cam_tensor = gradcam(input_tensor, class_idx=target_class_idx)
        cam_np = cam_tensor.squeeze().detach().cpu().numpy()
        images = blend_heatmap_with_image(pil_image, cam_np)
        return {
            "success": True,
            "images": images,
            "raw_cam": cam_np,
            "class_idx": target_class_idx,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "images": {
                "original": pil_image.convert("RGB"),
                "heatmap": pil_image.convert("RGB"),
                "overlay": pil_image.convert("RGB"),
            },
            "raw_cam": None,
            "class_idx": target_class_idx,
            "error": "Grad-CAM explanation could not be calculated.",
        }


def generate_demo_gradcam(pil_image: Image.Image, pathology: str = "Pneumonia") -> Dict[str, Any]:
    """
    Generates a synthetic anatomical Grad-CAM heatmap for DEMO MODE.
    Clearly simulated and grounded in general lung field geometry.
    """
    orig_rgb = pil_image.convert("RGB")
    w, h = orig_rgb.size

    # Create synthetic Gaussian attention in the mid/lower lung field
    y, x = np.ogrid[:h, :w]
    if "pneumonia" in pathology.lower() or "consolidation" in pathology.lower():
        center_x, center_y = int(w * 0.65), int(h * 0.55)  # Right lung base
        sigma_x, sigma_y = w * 0.18, h * 0.16
    elif "cardiomegaly" in pathology.lower():
        center_x, center_y = int(w * 0.50), int(h * 0.62)  # Cardiac silhouette
        sigma_x, sigma_y = w * 0.22, h * 0.18
    else:
        center_x, center_y = int(w * 0.50), int(h * 0.50)
        sigma_x, sigma_y = w * 0.25, h * 0.25

    dist = ((x - center_x) ** 2) / (2 * (sigma_x ** 2)) + ((y - center_y) ** 2) / (2 * (sigma_y ** 2))
    sim_cam = np.exp(-dist)

    images = blend_heatmap_with_image(orig_rgb, sim_cam)
    return {
        "success": True,
        "images": images,
        "raw_cam": sim_cam,
        "class_idx": 1,
        "error": None,
    }
