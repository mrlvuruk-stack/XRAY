"""
Ultrasound Explainability and Visualization Module
Handles segmentation mask overlays, boundary contour rendering, and classification CAM.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.cm as cm
import torch
from monai.visualize import GradCAM


def create_segmentation_overlay(
    pil_image: Image.Image,
    binary_mask: np.ndarray,
    fill_color: Tuple[int, int, int] = (0, 220, 130),     # Bright medical green
    contour_color: Tuple[int, int, int] = (255, 60, 60),  # Bright red contour
    alpha: float = 0.45
) -> Dict[str, Image.Image]:
    """
    Renders diagnostic segmentation outputs:
    1. Original sonogram
    2. Pure binary lesion mask
    3. Blended overlay with filled lesion area and highlighted contour boundary.
    """
    orig_rgb = pil_image.convert("RGB")
    width, height = orig_rgb.size

    # Resize mask to match original image dimensions
    mask_resized = cv2.resize(
        binary_mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # Pure binary mask image (black background, white lesion)
    mask_display = Image.fromarray((mask_resized * 255).astype(np.uint8)).convert("RGB")

    # Construct colored overlay
    orig_np = np.array(orig_rgb, dtype=np.float32)
    overlay_np = orig_np.copy()

    # Blend fill color where mask is active
    lesion_pixels = mask_resized > 0
    if np.any(lesion_pixels):
        color_layer = np.zeros_like(orig_np)
        color_layer[lesion_pixels] = fill_color
        overlay_np[lesion_pixels] = (
            (1.0 - alpha) * orig_np[lesion_pixels] + alpha * color_layer[lesion_pixels]
        )

        # Draw contour border using OpenCV
        contours, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        overlay_uint8 = np.clip(overlay_np, 0, 255).astype(np.uint8)
        cv2.drawContours(overlay_uint8, contours, -1, contour_color, 2)
        overlay_pil = Image.fromarray(overlay_uint8)
    else:
        overlay_pil = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))

    return {
        "original": orig_rgb,
        "mask": mask_display,
        "overlay": overlay_pil,
    }


def compute_ultrasound_gradcam(
    model: torch.nn.Module,
    target_layer: str,
    input_tensor: torch.Tensor,
    pil_image: Image.Image,
    target_class: Optional[int] = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Computes Grad-CAM for ultrasound classification."""
    if device is None:
        device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)
    target_idx = target_class if target_class is not None else 0

    try:
        gradcam = GradCAM(nn_module=model, target_layers=target_layer)
        cam_tensor = gradcam(input_tensor, class_idx=target_idx)
        cam_np = cam_tensor.squeeze().detach().cpu().numpy()

        orig_rgb = pil_image.convert("RGB")
        w, h = orig_rgb.size
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

        import matplotlib as mpl
        cmap = mpl.colormaps.get("jet", mpl.colormaps["viridis"])
        heat_np = np.uint8(255 * cmap(cam_norm)[:, :, :3])
        heat_pil = Image.fromarray(heat_np)
        overlay_np = (0.5 * heat_np + 0.5 * np.array(orig_rgb)).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        return {
            "success": True,
            "images": {
                "original": orig_rgb,
                "heatmap": heat_pil,
                "overlay": overlay_pil,
            },
            "error": None,
        }
    except Exception:
        return {
            "success": False,
            "images": {
                "original": pil_image.convert("RGB"),
                "heatmap": pil_image.convert("RGB"),
                "overlay": pil_image.convert("RGB"),
            },
            "error": "Ultrasound Grad-CAM calculation failed.",
        }


def generate_demo_ultrasound_segmentation(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Generates a realistic simulated ultrasound lesion segmentation for DEMO MODE.
    Detects hypoechoic center or creates a typical sonographic nodule shape.
    """
    orig_rgb = pil_image.convert("RGB")
    w, h = orig_rgb.size
    np_gray = np.array(orig_rgb.convert("L"))

    # Create an elliptical lesion mask in center-lower acoustic window
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w * 0.48), int(h * 0.52))
    axes = (int(w * 0.16), int(h * 0.12))
    angle = -15
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)

    # Add realistic boundary roughness
    noise = np.random.RandomState(42).normal(0, 1, (h, w))
    smooth_noise = cv2.GaussianBlur(noise, (15, 15), 0)
    rough_mask = ((mask > 0) & (smooth_noise > -0.4)).astype(np.uint8)
    if np.sum(rough_mask) == 0:
        rough_mask = mask

    res_images = create_segmentation_overlay(orig_rgb, rough_mask)
    lesion_pixels = int(np.sum(rough_mask))
    area_pct = round((lesion_pixels / rough_mask.size) * 100, 2)

    return {
        "images": res_images,
        "binary_mask": rough_mask,
        "lesion_pixels": lesion_pixels,
        "lesion_area_percentage": area_pct,
    }


def generate_demo_ultrasound_classification(pil_image: Image.Image) -> Dict[str, Any]:
    """Generates simulated classification CAM for sonography demo."""
    orig_rgb = pil_image.convert("RGB")
    w, h = orig_rgb.size

    y, x = np.ogrid[:h, :w]
    cx, cy = int(w * 0.50), int(h * 0.50)
    dist = ((x - cx) ** 2) / (2 * ((w * 0.2) ** 2)) + ((y - cy) ** 2) / (2 * ((h * 0.2) ** 2))
    sim_cam = np.exp(-dist)

    import matplotlib as mpl
    cmap = mpl.colormaps.get("magma", mpl.colormaps["viridis"])
    heat_np = np.uint8(255 * cmap(sim_cam)[:, :, :3])

    heat_pil = Image.fromarray(heat_np)
    overlay_np = (0.5 * heat_np + 0.5 * np.array(orig_rgb)).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_np)

    return {
        "images": {
            "original": orig_rgb,
            "heatmap": heat_pil,
            "overlay": overlay_pil,
        }
    }
