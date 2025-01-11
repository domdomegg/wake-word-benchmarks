#!/usr/bin/env python3
import wave
import numpy as np
from pathlib import Path
from tqdm import tqdm
from models import MicroWakeWordModel

class WakeWordBenchmark:
    def __init__(self, model, pos_sample_dir: str = 'alexa', neg_sample_dir: str = 'common_voice', max_samples: int = 100):
        self.model = model
        self.frame_size = 1280  # 80ms at 16kHz
        self.results = self._run_benchmark(
            pos_sample_dir=pos_sample_dir,
            neg_sample_dir=neg_sample_dir,
            max_samples=max_samples
        )
    
    def _test_file(self, audio_path: Path):
        """Returns True if wake word detected in file."""
        with wave.open(str(audio_path), mode='rb') as f:
            # Verify sample rate
            if f.getframerate() != 16000:
                raise ValueError(f"Audio must be 16kHz, got {f.getframerate()}Hz")
            
            # Load WAV clip frames
            audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        
        # Process audio with increasing windows
        predictions = self.model.predict_clip(audio)
        if max(predictions) > 0.9:
            return True
        return False

    def _run_benchmark(self, pos_sample_dir, neg_sample_dir, max_samples):
        """Run benchmark and return basic metrics."""
        pos_samples = list(Path('positive_samples/' + pos_sample_dir).glob('**/*.wav'))[:max_samples]
        neg_samples = list(Path('negative_samples/' + neg_sample_dir).glob('**/*.wav'))[:max_samples]
        
        false_rejects = sum(not self._test_file(f) for f in tqdm(pos_samples, desc='Testing positive samples'))
        false_accepts = sum(self._test_file(f) for f in tqdm(neg_samples, desc='Testing negative samples'))
        
        return {
            'false_reject_rate': false_rejects / len(pos_samples),
            'false_accept_rate': false_accepts / len(neg_samples)
        }

if __name__ == '__main__':
    print('Starting benchmark...')
    benchmark = WakeWordBenchmark(MicroWakeWordModel('alexa'), 'alexa')
    print(benchmark.results)
