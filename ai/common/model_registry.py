"""
Model Registry Module
Provides an extensible registry architecture for medical imaging models (X-Ray, Ultrasound, and future CT/MRI).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from PIL import Image
import torch


class BaseMedicalModel(ABC):
    """
    Abstract Base Class for all medical imaging models.
    Every modality (X-Ray, Ultrasound, CT, MRI) must implement this interface.
    """

    def __init__(
        self,
        name: str,
        version: str,
        modality: str,
        task: str,
        labels: List[str],
        weights_path: Optional[str] = None
    ):
        self.name = name
        self.version = version
        self.modality = modality.lower()  # xray, ultrasound, mri, ct
        self.task = task.lower()          # classification, segmentation
        self.labels = labels
        self.weights_path = weights_path
        self._model: Optional[torch.nn.Module] = None
        self._is_loaded: bool = False

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if actual model weights are available on disk and ready to load."""
        pass

    @abstractmethod
    def load_weights(self, device: torch.device) -> bool:
        """Loads actual model weights onto the target device."""
        pass

    @abstractmethod
    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """Applies MONAI transforms to convert PIL image to a normalized tensor."""
        pass

    @abstractmethod
    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Runs model forward pass and returns findings/probabilities or segmentation maps."""
        pass

    @abstractmethod
    def explain(
        self,
        pil_image: Image.Image,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Computes explainability artifact (e.g. Grad-CAM, attention map, or segmentation overlay)."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Returns model specification metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "modality": self.modality,
            "task": self.task,
            "labels": self.labels,
            "configured": self.is_configured(),
            "weights_path": self.weights_path,
        }


class ModelRegistry:
    """
    Central Singleton registry managing all registered medical imaging models.
    Allows easy dynamic registration of new modalities.
    """
    _instance: Optional["ModelRegistry"] = None
    _registry: Dict[str, BaseMedicalModel] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._registry = {}
        return cls._instance

    @classmethod
    def register(cls, model: BaseMedicalModel) -> None:
        """Registers a medical model into the registry."""
        key = f"{model.modality}:{model.task}:{model.name}".lower()
        cls._registry[key] = model

    @classmethod
    def get_model(cls, modality: str, task: str = "classification") -> Optional[BaseMedicalModel]:
        """Retrieves a model matching modality and task."""
        modality = modality.lower()
        task = task.lower()
        for key, model in cls._registry.items():
            if model.modality == modality and model.task == task:
                return model
        return None

    @classmethod
    def list_models(cls, modality: Optional[str] = None) -> List[BaseMedicalModel]:
        """Lists all registered models, optionally filtered by modality."""
        if modality:
            modality = modality.lower()
            return [m for m in cls._registry.values() if m.modality == modality]
        return list(cls._registry.values())

    @classmethod
    def get_status_overview(cls) -> Dict[str, Dict[str, Any]]:
        """Returns a summary of all registered models and their configuration status."""
        overview = {}
        for key, model in cls._registry.items():
            overview[key] = {
                "name": model.name,
                "version": model.version,
                "modality": model.modality,
                "task": model.task,
                "configured": model.is_configured(),
                "labels_count": len(model.labels),
            }
        return overview

    @classmethod
    def clear(cls) -> None:
        """Clears the registry (useful for testing)."""
        cls._registry.clear()
