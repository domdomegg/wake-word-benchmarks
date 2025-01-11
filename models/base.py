import numpy as np
from abc import ABC, abstractmethod

class WakeWordModel(ABC):
    """Abstract base class for wake word detection models."""
    
    @abstractmethod
    def predict_clip(self, clip: np.ndarray) -> list[float]:
        """Process frames of audio and return predictions.
        
        Args:
            clip: Raw audio samples as int16 numpy array at 16kHz.
            
        Returns:
            List of prediction scores between 0 and 1 for each processed window.
        """
        pass
