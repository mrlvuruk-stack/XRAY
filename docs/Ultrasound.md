# Ultrasound AI Diagnostic Pipeline

## 1. Overview
The Ultrasound diagnostic engine operates as an independent pipeline specifically tailored to B-mode acoustic sonography. It is completely decoupled from the Chest X-Ray pipeline and supports two distinct medical computer vision tasks:

1. **Pathology Classification:** Evaluates lesions into Normal, Benign Lesion, or Malignant Finding.
2. **Lesion Segmentation:** Delineates hypoechoic or hyperechoic lesion boundaries using MONAI UNet.

## 2. Pipeline Architectures

### Task A: Lesion Segmentation (MONAI UNet)
```
Upload Ultrasound Scan
       ↓
Image Validation (Aspect ratio, pixel variance, file size)
       ↓
MONAI Preprocessing (256x256 resizing, intensity normalization)
       ↓
MONAI UNet Architecture (spatial_dims=2, 4-stage encoder-decoder)
       ↓
Sigmoid Activation & Thresholding (≥ 0.50 binary lesion mask)
       ↓
Quantification (Lesion Pixel Count, Lesion Area Percentage)
       ↓
Visualization (Original, Binary Mask, Blended Overlay + Red Contour)
       ↓
ReportLab PDF Generation
```

### Task B: Pathology Classification (MONAI DenseNet121)
```
Upload Ultrasound Scan
       ↓
Validation & MONAI Preprocessing
       ↓
DenseNet121 Classifier (3 classes: Normal, Benign, Malignant)
       ↓
Softmax Multi-Class Probabilities
       ↓
Grad-CAM Activation Heatmap
       ↓
Visualization & PDF Export
```

## 3. Acoustic Speckle & Preprocessing
Ultrasound images feature characteristic granular speckle patterns caused by constructive and destructive interference of backscattered acoustic waves. The MONAI pipeline normalizes intensity ranges while preserving acoustic interface edges.

## 4. Architectural Decoupling
The ultrasound pipeline does not share weights, preprocessing dimensions, or models with the X-ray pipeline, ensuring modality-specific feature representation and preventing cross-domain feature corruption.
