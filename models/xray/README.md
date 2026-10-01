# Chest X-Ray Model Weights Directory

This directory stores real deep-learning model weights for the Chest X-Ray diagnostic pipeline.

## Default Weights File

Place your trained PyTorch/MONAI checkpoint here:
```
models/xray/chexnet_monai_densenet121.pth
```

## Model Architecture Specifications

- **Backbone:** MONAI `DenseNet121` (`monai.networks.nets.DenseNet121`)
- **Spatial Dimensions:** `2D` (`spatial_dims=2`)
- **Input Channels:** `1` (Single-channel grayscale radiograph)
- **Output Channels:** `8`
- **Output Pathology Classes:**
  1. `Normal`
  2. `Pneumonia`
  3. `Pleural Effusion`
  4. `Atelectasis`
  5. `Cardiomegaly`
  6. `Infiltration`
  7. `Consolidation`
  8. `Nodule`

## How to Configure Real Weights

1. Obtain a trained MONAI DenseNet121 checkpoint (e.g. trained on NIH ChestX-ray14, CheXpert, or PadChest).
2. Save the PyTorch `state_dict` to `models/xray/chexnet_monai_densenet121.pth`:
   ```python
   import torch
   from monai.networks.nets import DenseNet121

   model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=8)
   # ... after training or downloading weights ...
   torch.save(model.state_dict(), "models/xray/chexnet_monai_densenet121.pth")
   ```
3. Set `DEMO_MODE=false` in `.env` or toggle it off in the application sidebar.
4. The system will detect the weights and switch from `MODEL NOT CONFIGURED` to `REAL PRETRAINED MODEL`.
