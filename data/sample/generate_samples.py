"""
SYNTHETIC TECHNICAL PIPELINE TEST DATA GENERATOR
------------------------------------------------
WARNING: These generated images are STRICTLY synthetic technical test arrays
created for verifying MONAI transform pipelines, resizing, intensity scaling,
and tensor shapes.
- They are NOT clinical patient scans.
- They do NOT represent real human anatomy or diagnostic cases.
- DEMO DATA — NOT FOR CLINICAL USE.
"""


import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2


def generate_chest_xray(has_consolidation: bool = False) -> Image.Image:
    """
    SYNTHETIC TECHNICAL TEST DATA ONLY:
    Generates a 2D synthetic radiograph-like numpy test array strictly for testing MONAI
    transform pipelines, resizing, intensity scaling, and tensor shapes.
    NOT a clinical patient scan. DEMO DATA — NOT FOR CLINICAL USE.
    """
    size = (512, 512)
    img = np.zeros(size, dtype=np.float32)

    # Base background soft tissue density
    y, x = np.ogrid[:size[0], :size[1]]
    cx, cy = size[1] // 2, size[0] // 2

    # Mediastinum and spine (high density / bright)
    spine = np.exp(-((x - cx) ** 2) / (2 * (30 ** 2))) * 140
    img += spine

    # Bilateral lung fields (radiolucent / darker)
    left_lung_dist = ((x - (cx - 100)) ** 2) / (2 * (65 ** 2)) + ((y - (cy - 10)) ** 2) / (2 * (130 ** 2))
    right_lung_dist = ((x - (cx + 100)) ** 2) / (2 * (65 ** 2)) + ((y - (cy - 10)) ** 2) / (2 * (130 ** 2))
    lung_field = (np.exp(-left_lung_dist) + np.exp(-right_lung_dist)) * 110
    img += (180 - lung_field)

    # Cardiac silhouette (left lower mediastinum density)
    heart_dist = ((x - (cx - 40)) ** 2) / (2 * (75 ** 2)) + ((y - (cy + 70)) ** 2) / (2 * (60 ** 2))
    heart = np.exp(-heart_dist) * 90
    img += heart

    # Rib cage arches
    for rib_y in range(100, 440, 42):
        rib_arch = np.exp(-((y - (rib_y + 0.05 * (x - cx) ** 2 / 20)) ** 2) / (2 * (6 ** 2))) * 35
        img += rib_arch

    # Diaphragm domes
    diaphragm_left = np.exp(-((y - 420) ** 2) / (2 * (25 ** 2))) * (x < cx) * 70
    diaphragm_right = np.exp(-((y - 410) ** 2) / (2 * (25 ** 2))) * (x >= cx) * 70
    img += diaphragm_left + diaphragm_right

    # If pneumonia demo: add patchy alveolar consolidation in right lower lung field
    if has_consolidation:
        consol_dist = ((x - (cx + 90)) ** 2) / (2 * (45 ** 2)) + ((y - (cy + 75)) ** 2) / (2 * (35 ** 2))
        patchy_noise = np.random.RandomState(101).normal(1.0, 0.2, size)
        consolidation = np.exp(-consol_dist) * 110 * patchy_noise
        img += consolidation

    # Add realistic anatomical X-ray quantum noise
    quantum_noise = np.random.RandomState(42).normal(0, 4, size)
    img += quantum_noise

    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def generate_ultrasound_scan(has_lesion: bool = False) -> Image.Image:
    """
    SYNTHETIC TECHNICAL TEST DATA ONLY:
    Generates a 2D synthetic B-mode ultrasound-like test array strictly for verifying
    MONAI speckle filtering, resizing, and normalization pipelines.
    NOT a clinical patient scan. DEMO DATA — NOT FOR CLINICAL USE.
    """
    size = (384, 384)
    h, w = size

    # Rayleigh speckle noise background
    speckle = np.random.RandomState(7).rayleigh(scale=35, size=size)

    # Tissue depth attenuation gradient (darker near bottom)
    depth_gradient = np.linspace(1.2, 0.6, h)[:, None]
    img = speckle * depth_gradient

    # Layered acoustic interfaces (fascial layers / stratum)
    for layer_y in [90, 160, 250]:
        layer = np.exp(-((np.arange(h)[:, None] - layer_y) ** 2) / (2 * (4 ** 2))) * 60
        img += layer

    # Fan beam or trapezoidal field mask
    y, x = np.ogrid[:h, :w]
    sector_mask = (y > 30) & (y < 360) & (x > 40) & (x < 344)
    img = img * sector_mask

    # If lesion demo: hypoechoic nodule with acoustic shadow
    if has_lesion:
        cx, cy = int(w * 0.52), int(h * 0.50)
        lesion_dist = ((x - cx) ** 2) / (2 * (42 ** 2)) + ((y - cy) ** 2) / (2 * (28 ** 2))
        lesion_mask = lesion_dist < 1.0
        # Dark hypoechoic interior
        img[lesion_mask] = img[lesion_mask] * 0.30
        # Posterior acoustic enhancement or shadow
        shadow_mask = (x > (cx - 35)) & (x < (cx + 35)) & (y > (cy + 25)) & (y < 350)
        img[shadow_mask] = img[shadow_mask] * 0.70

    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def generate_all_samples(output_dir: str):
    """Generates synthetic technical test data for pipeline testing."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    xray_normal = generate_chest_xray(has_consolidation=False)
    xray_normal.save(out_path / "xray_normal.png")

    xray_pneumonia = generate_chest_xray(has_consolidation=True)
    xray_pneumonia.save(out_path / "xray_pneumonia.png")

    us_normal = generate_ultrasound_scan(has_lesion=False)
    us_normal.save(out_path / "ultrasound_normal.png")

    us_lesion = generate_ultrasound_scan(has_lesion=True)
    us_lesion.save(out_path / "ultrasound_lesion.png")

    print(f"Synthetic technical pipeline test data successfully saved to {out_path}!")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    generate_all_samples(str(current_dir))
