import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod
import openwakeword
from openwakeword.model import Model as OWWModel
from openwakeword.utils import download_models

class OpenWakeWordModel:
    """Simple wrapper for OpenWakeWord model."""
    
    def __init__(self, model_name):
        print("Downloading pre-trained " + model_name + " model...")
        download_models(model_names=[model_name])
        print("Starting pre-trained " + model_name + " model...")
        self.model_name = model_name
        self.model = OWWModel(inference_framework="onnx", wakeword_models=[model_name])
    
    def predict_clip(self, clip: np.ndarray) -> list[float]:
        """Process frames of a wav file."""
        self.model.reset()
        predictions = self.model.predict_clip(clip)
        return [p[self.model_name] for p in predictions]
