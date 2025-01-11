import numpy as np
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import openwakeword
from openwakeword.model import Model as OWWModel
from openwakeword.utils import download_models
import tensorflow as tf
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
import os
import urllib.request

class OpenWakeWordModel:    
    def __init__(self, model_name: str):
        print("Downloading pre-trained " + model_name + " model (if not present)...")
        download_models(model_names=[model_name])
        print("Starting pre-trained " + model_name + " model...")
        self.model_name = model_name
        self.model = OWWModel(inference_framework="onnx", wakeword_models=[model_name])
    
    def predict_clip(self, clip: np.ndarray) -> list[float]:
        """Process frames of a wav file."""
        self.model.reset()
        predictions = self.model.predict_clip(clip)
        return [p[self.model_name] for p in predictions]

class MicroWakeWordModel:    
    def __init__(self, model_name: str = 'alexa'):
        """Initialize the model with the given TFLite model path.
        
        Args:
            model_path: Path to the TFLite model file. If using the default model and
                       it doesn't exist, it will be downloaded automatically.
        """
        self.model_name = model_name
        self.model_path = os.path.abspath(os.path.join('models', 'microWakeWord', model_name + '.tflite'))
        self._download_model()
        
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.is_quantized = self.input_details['dtype'] == np.int8
    
    def _download_model(self) -> None:
        """Attempt to download the model if not present."""
        if not os.path.exists(self.model_path):
            print("Downloading model " + self.model_name + ".tflite...")
            url = "https://github.com/esphome/micro-wake-word-models/raw/refs/heads/main/models/v2/" + self.model_name + ".tflite"
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete")
    
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process raw audio into features using the microfrontend.
        
        Args:
            audio: Raw audio samples as int16 numpy array.
            
        Returns:
            Processed features as float32 numpy array.
        """
        # Normalize and clip audio
        audio = np.clip(audio.astype(np.float32) / 32768.0 * 32768, -32768, 32767).astype(np.int16)
        
        # Generate features using microfrontend
        features = frontend_op.audio_microfrontend(
            tf.convert_to_tensor(audio),
            sample_rate=16000,
            window_size=30,
            window_step=20,
            num_channels=40,
            upper_band_limit=7500,
            lower_band_limit=125,
            enable_pcan=True,
            min_signal_remaining=0.05,
            out_scale=1,
            out_type=tf.uint16,
        ).numpy().astype(np.float32) * 0.0390625
        
        return features
    
    def predict_clip(self, clip: np.ndarray) -> list[float]:
        """Process frames of audio and return predictions.
        
        Args:
            clip: Raw audio samples as int16 numpy array at 16kHz.
            
        Returns:
            List of prediction scores between 0 and 1 for each processed window.
        """
        # Process audio into features
        features = self._process_audio(clip)
        
        predictions = []
        # Process features in sliding windows of 3 frames
        for i in range(len(features) - 2):
            chunk = np.reshape(features[i:i+3], (1, 3, 40))
            
            # Handle quantization if model is quantized
            if self.is_quantized:
                input_scale = self.input_details['quantization_parameters']['scales'][0]
                input_zero_point = self.input_details['quantization_parameters']['zero_points'][0]
                chunk = (chunk / input_scale + input_zero_point).astype(np.int8)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details["index"], chunk)
            self.interpreter.invoke()
            
            # Get and process output
            output = self.interpreter.get_tensor(self.output_details["index"])[0][0]
            if self.is_quantized:
                zero_point = self.output_details['quantization_parameters']['zero_points'][0]
                output = (output.astype(np.float32) - zero_point) / 255.0
            
            predictions.append(float(output))
        
        return predictions
