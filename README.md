# SWASTHYA MONAI MEDICAL IMAGING LAB
### Standalone Multi-Modal Medical AI Diagnostic Platform powered by MONAI & PyTorch

---

## 1. Project Overview

**Swasthya MONAI Medical Imaging Lab** is a completely standalone, privacy-first healthcare AI diagnostic screening and visualization platform. Built natively with **MONAI (Medical Open Network for AI)** and **PyTorch**, it provides clinical decision support capabilities across multiple radiological modalities:

1. **Chest X-Ray AI:** Multi-finding radiographic pathology detection (Pneumonia, Cardiomegaly, Pleural Effusion, Atelectasis, Infiltration, Consolidation, Nodule, Normal) with **MONAI Grad-CAM** spatial explainability.
2. **Ultrasound AI:** Sonographic acoustic computing supporting dual clinical tasks:
   - **Pathology Classification:** Tri-class lesion classification (Normal vs Benign vs Malignant).
   - **Lesion Segmentation:** Precise contour and area boundary extraction using **MONAI UNet**.
3. **Modular Architecture:** Pre-structured for upcoming 3D volumetric modalities (**CT** & **MRI**).

> [!IMPORTANT]
> **Strict Clinical Integrity Standards:** This system never invents model weights, accuracy metrics, or medical diagnoses. It enforces three distinct, unambiguous states:
> - `REAL MODEL`: Inference executed using verified deep learning weights.
> - `DEMO MODE`: Clearly labeled simulation (`DEMO RESULT — NOT A REAL MEDICAL PREDICTION`).
> - `MODEL NOT CONFIGURED`: Informative status with step-by-step weight configuration instructions.

---

## 2. Architecture & Design Principles

```
swasthya-monai-imaging/
│
├── app/                                 # Presentation Layer (Streamlit)
│   ├── main.py                          # Application entry point & navigation
│   ├── ui/
│   │   ├── dashboard.py                 # Multi-modality diagnostic overview & device telemetry
│   │   ├── xray_page.py                 # Chest X-Ray screening & Grad-CAM viewer
│   │   └── ultrasound_page.py           # Sonography classification & UNet segmentation
│   └── components/
│       ├── image_viewer.py              # Medical image windowing & metadata inspector
│       ├── prediction_card.py           # Findings table & risk stratification badges
│       ├── heatmap_viewer.py            # [Original] [Heatmap] [Overlay] visualization
│       └── report_viewer.py             # ReportLab clinical PDF export
│
├── ai/                                  # Core AI & Medical Engineering Layer
│   ├── common/
│   │   ├── device.py                    # CUDA / CPU automatic hardware detection
│   │   ├── image_validation.py          # Format, resolution, integrity & path security
│   │   └── model_registry.py            # BaseMedicalModel & central ModelRegistry
│   ├── xray/
│   │   ├── config.py                    # Pathology labels, input shapes, alert thresholds
│   │   ├── preprocessing.py             # MONAI radiograph transform pipeline
│   │   ├── model.py                     # MONAI DenseNet121 neural network architecture
│   │   ├── inference.py                 # End-to-end execution pipeline
│   │   └── explainability.py            # MONAI GradCAM convolutional feature attribution
│   └── ultrasound/
│       ├── config.py                    # Sonography labels, tasks, resolutions
│       ├── preprocessing.py             # MONAI acoustic speckle normalization
│       ├── model.py                     # MONAI DenseNet121 (Cls) + MONAI UNet (Seg)
│       ├── inference.py                 # Dual-task sonography execution pipeline
│       └── explainability.py            # UNet boundary overlays & classification CAM
│
├── models/                              # Local Model Weight Storage
│   ├── xray/
│   │   └── README.md                    # X-ray weight placement guide
│   └── ultrasound/
│       └── README.md                    # Ultrasound classification & UNet weight guide
│
├── data/
│   ├── sample/                          # Bundled anatomical validation scans
│   ├── uploads/                         # Temporary local upload cache
│   └── outputs/                         # Output visualizations
│
├── reports/                             # Generated ReportLab clinical PDF reports
│
├── tests/                               # Automated Test Suite (18 tests)
│   ├── test_preprocessing.py            # MONAI transform validation
│   ├── test_image_validation.py         # Format, corruption & security checks
│   ├── test_xray.py                     # X-ray pipeline, demo mode & unconfigured states
│   ├── test_ultrasound.py               # Ultrasound dual-task pipeline tests
│   └── test_report.py                   # ReportLab PDF generation verification
│
├── docs/                                # Technical & Clinical Documentation
│   ├── XRay.md                          # Radiographic pipeline documentation
│   └── Ultrasound.md                    # Sonography pipeline documentation
│
├── .env.example                         # Environment configuration template
├── .env                                 # Local configuration (DEMO_MODE=true)
├── requirements.txt                     # Minimal Python dependencies
├── README.md                            # Comprehensive documentation
└── run.py                               # Application runner script
```

---

## 3. Technology Stack

- **Python:** 3.10 - 3.13 supported
- **MONAI:** `v1.6.0` (Medical Open Network for AI - transforms, DenseNet121, UNet, GradCAM)
- **PyTorch:** `v2.1.0+` (Deep learning tensor execution engine)
- **Streamlit:** Modern medical laboratory web interface
- **OpenCV & Pillow:** Image processing & boundary contour detection
- **Matplotlib:** Heatmap color mapping (Jet, Magma)
- **ReportLab:** Standardized clinical diagnostic PDF generation
- **PyTest:** Comprehensive test suite

---

## 4. Installation & Setup

### Prerequisites
Ensure Python 3.10+ is installed on your system.

### Step 1: Clone or Navigate to Project
```bash
cd swasthya-monai-imaging
```

### Step 2: Install PyTorch & MONAI
```bash
# CPU Mode (Recommended for testing / hackathons)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU Mode (If NVIDIA CUDA is available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install MONAI and platform dependencies
pip install -r requirements.txt
```

---

## 5. Running the Application

Launch the platform using the single runner command:
```bash
python run.py
```
Or directly with Streamlit:
```bash
streamlit run app/main.py
```

The application will be accessible at: `http://localhost:8501`

---

## 6. How the Modalities Work

### 🫁 How Chest X-Ray AI Works
1. **Upload & Sanitize:** Accepts PNG, JPG, TIFF, or DICOM. Validates dimensions (min 32x32, max 8192x8192), checks pixel variance to reject blank scans, and strips directory paths.
2. **MONAI Preprocessing:**
   - `EnsureChannelFirst(channel_dim="no_channel")` standardizes grayscale format `(1, H, W)`.
   - `Resize((224, 224))` formats image to DenseNet receptive field.
   - `ScaleIntensityRange(0, 255, 0.0, 1.0)` normalizes dynamic range.
3. **Model Prediction:** MONAI `DenseNet121` evaluates 8 co-occurring findings with sigmoid multi-label probabilities:
   - Normal, Pneumonia, Pleural Effusion, Atelectasis, Cardiomegaly, Infiltration, Consolidation, Nodule.
4. **Grad-CAM Explainability:** Computes gradients at `features.denseblock4.denselayer16.layers.conv2`, projects heatmaps, and blends an alpha overlay with the radiograph.
5. **ReportLab PDF Export:** Compiles findings, overlays, and disclaimers into a downloadable PDF report.

### 🔬 How Ultrasound AI Works
1. **Decoupled Architecture:** Operates with separate transforms and model weights tailored to B-mode acoustic scans.
2. **Task Selection:**
   - **Classification Mode:** MONAI `DenseNet121` predicts `Normal`, `Benign Lesion`, or `Malignant Finding` with Grad-CAM heatmap visualization.
   - **Segmentation Mode:** MONAI `UNet` (`spatial_dims=2`, `channels=(16, 32, 64, 128)`) computes pixel-level lesion boundary masks, calculates lesion area percentage, and renders green alpha fill with red boundary contours.
3. **PDF Export:** Embeds binary mask and contour overlay in clinical report.

---

## 7. How to Add Real Model Weights

The platform will display `"Chest X-ray model is not configured."` or `"Ultrasound model is not configured."` when weights are absent and `DEMO_MODE=false`.

### Adding Chest X-Ray Weights
1. Train or download a MONAI `DenseNet121` checkpoint with 8 output classes (e.g. from NIH ChestX-ray14 / CheXNet).
2. Save the `.pth` file to:
   ```
   models/xray/chexnet_monai_densenet121.pth
   ```
3. Set `DEMO_MODE=false` in `.env` or toggle Demo Mode off in the UI sidebar.

### Adding Ultrasound Weights
1. **Classification:**
   Save checkpoint with 3 classes to:
   ```
   models/ultrasound/classification/ultrasound_densenet121.pth
   ```
2. **Segmentation:**
   Save MONAI UNet checkpoint to:
   ```
   models/ultrasound/segmentation/ultrasound_unet.pth
   ```

---

## 8. Demo Mode vs Real Model vs Model Not Configured

| Status | Trigger Condition | Visual Appearance | Diagnostic Output |
| :--- | :--- | :--- | :--- |
| **REAL MODEL** | Weights file exists on disk & `DEMO_MODE=false` | Green badge: `● REAL PRETRAINED MODEL` | Real forward-pass neural network predictions |
| **DEMO MODE** | `DEMO_MODE=true` in `.env` or sidebar toggle | Amber warning: `⚠️ DEMO RESULT — NOT A REAL MEDICAL PREDICTION` | Realistic simulated findings for UI/testing workflows |
| **NOT CONFIGURED**| Weights missing & `DEMO_MODE=false` | Red alert: `✖ MODEL NOT CONFIGURED` | Predictions blocked; instructions displayed |

---

## 9. Future Extension Path for CT and MRI

The architecture is built on the `BaseMedicalModel` abstract base class and `ModelRegistry`:

```
               ┌───────────────────────┐
               │   BaseMedicalModel    │
               └───────────┬───────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────┴────────┐┌───────┴───────┐┌────────┴────────┐
│  XRayDenseNet   ││ UltrasoundNet ││  Future MRINet  │
│    (2D MONAI)   ││ (DenseNet/UNet││ (3D SwinUNETR)  │
└─────────────────┘└───────────────┘└─────────────────┘
```

To add **MRI** or **CT**:
1. Create `ai/mri/` with `preprocessing.py`, `model.py`, and `inference.py`.
2. Implement MONAI 3D transforms (`EnsureChannelFirst3D`, `Spacingd`, `Orientationd`).
3. Inherit from `BaseMedicalModel` and register with `ModelRegistry.register(MRIModel())`.
4. Add `app/ui/mri_page.py` with multi-slice sagittal, coronal, and axial slice viewers.

---

## 10. Running Automated Tests

Run the full pytest suite:
```bash
python -m pytest tests -v
```

The test suite covers:
- Image validation (valid PNG/JPG, corrupted bytes, empty files, solid blank scans, filename sanitization)
- MONAI preprocessing pipelines (PIL and NumPy tensor transformations, dimension and range checks)
- X-ray inference (unconfigured state, demo mode, Grad-CAM generation)
- Ultrasound inference (classification and segmentation pipelines, mask generation, demo/unconfigured states)
- ReportLab PDF generation (valid PDF byte stream, layout structure, medical disclaimer inclusion)

---

## 11. Privacy & Security

- **Strictly Local:** Uploaded medical images are processed entirely on your local machine.
- **No External Cloud APIs:** Zero images or patient metadata are transmitted to external servers.
- **Path Disclosure Prevention:** File paths and system internals are sanitized before presentation.

---

## 12. Medical Disclaimer

> [!CAUTION]
> **MANDATORY MEDICAL DISCLAIMER:**
> AI-generated output is intended for clinical decision support and research/prototype purposes. It is not a definitive diagnosis and must be reviewed by a qualified healthcare professional..
