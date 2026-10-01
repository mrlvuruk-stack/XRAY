# Ultrasound Model Weights Directory

This directory stores weights for the dedicated Ultrasound sonography pipeline.
Note: The Ultrasound pipeline is completely decoupled from the Chest X-Ray pipeline.

## Subdirectories

### 1. Classification
Path:
```
models/ultrasound/classification/ultrasound_densenet121.pth
```
- **Architecture:** MONAI `DenseNet121` (`spatial_dims=2, in_channels=1, out_channels=3`)
- **Classes:**
  1. `Normal`
  2. `Benign Lesion`
  3. `Malignant Finding`

### 2. Segmentation
Path:
```
models/ultrasound/segmentation/ultrasound_unet.pth
```
- **Architecture:** MONAI `UNet` (`spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2)`)
- **Output:** Binary lesion contour and area mask.

## How to Add Model Weights

```python
import torch
from monai.networks.nets import DenseNet121, UNet

# 1. Classification
cls_model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
torch.save(cls_model.state_dict(), "models/ultrasound/classification/ultrasound_densenet121.pth")

# 2. Segmentation
seg_model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2))
torch.save(seg_model.state_dict(), "models/ultrasound/segmentation/ultrasound_unet.pth")
```
