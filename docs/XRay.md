# Chest X-Ray AI Diagnostic Pipeline

## 1. Overview
The Chest X-Ray diagnostic engine is built on **MONAI (Medical Open Network for AI)** and **PyTorch**, designed to assist radiologists and healthcare providers by screening chest radiographs for common cardiopulmonary pathologies.

## 2. Pipeline Architecture

```
Upload Radiograph
       ↓
Image Validation (Format, dimensions, corruptions, intensity check)
       ↓
MONAI Preprocessing (Channel formatting, Bilinear resize to 224x224, ScaleIntensityRange [0, 1])
       ↓
DenseNet121 Model (8-finding multi-label output)
       ↓
Sigmoid Probabilities & Risk Stratification (High ≥ 50%, Moderate ≥ 25%, Low < 25%)
       ↓
MONAI GradCAM (Features denseblock4 conv2 layer activation)
       ↓
Interactive Visualization (Original, Jet Heatmap, Alpha Blended Overlay)
       ↓
ReportLab PDF Generation (Clinical metadata, side-by-side scans, disclaimer)
```

## 3. MONAI Transforms
Medical imaging standardizes raw DICOM/PNG pixel distributions via `monai.transforms`:
- `EnsureChannelFirst(channel_dim="no_channel")`: Enforces single-channel grayscale structure `(1, H, W)`.
- `Resize(spatial_size=(224, 224), mode="bilinear", anti_aliasing=True)`: Preserves aspect ratio while matching DenseNet receptive field.
- `ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True)`: Rescales uint8 pixel values to float32 normalized intensities.
- `EnsureType(data_type="tensor", dtype=torch.float32)`: Yields PyTorch MetaTensor ready for device execution.

## 4. Multi-Label Classification
Chest radiographs frequently exhibit co-occurring pathologies (e.g. Pneumonia with Pleural Effusion). Sigmoid activation allows independent probability evaluation across all 8 target findings:
1. Normal
2. Pneumonia
3. Pleural Effusion
4. Atelectasis
5. Cardiomegaly
6. Infiltration
7. Consolidation
8. Nodule

## 5. Explainability (Grad-CAM)
Using `monai.visualize.GradCAM`:
- **Target Layer:** `features.denseblock4.denselayer16.layers.conv2`
- **Output:** Gradients of the target class score with respect to feature activation maps are pooled to generate a coarse 2D localization map highlighting anatomical regions of concern (e.g., lower lung consolidation or cardiomegaly).

## 6. Safety & Privacy
- **100% Local Inference:** No scan leaves the machine.
- **Strict Real vs Demo Separation:** If weights are missing, predictions are blocked with a clear warning instead of inventing data. Demo mode explicitly watermarks outputs.
