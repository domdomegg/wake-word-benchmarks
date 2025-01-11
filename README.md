# Wake Word Benchmark

Benchmark tool to compare the performance of openWakeWord wake word detection systems.

## Requirements

- ffmpeg (for audio conversion)
- Python 3.8-3.11 (for Tensorflow support)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Download test data
python download_test_data.py

# Run benchmark
python benchmark.py
```

## Test Data

The benchmark uses real test data from established sources:

1. **Wake Word Samples**: Recordings from Picovoice's wake-word-benchmark, from [my pre-converted repository](https://github.com/domdomegg/picovoice-wake-word-benchmark)
   - Multiple speakers
   - Various wake words (alexa, computer, jarvis, etc.)
   - Clean recordings

2. **Negative Samples**: [CommonVoice](https://commonvoice.mozilla.org/) dataset
   - 45 hours of crowd-sourced speech recordings
   - Diverse speakers and accents
   - Various topics and speaking styles

## What it Tests

1. **False Accept Rate**: How often the system incorrectly activates
2. **False Reject Rate**: How often it misses the wake word

## Adding New Models

The benchmark can be extended to test additional wake word models. To do this:

1. Create a new file in the `models/` directory (e.g. `models/my_model.py`)
2. Implement a class that inherits from [`WakeWordModel`](./models/base.py).
   This includes the `predict_clip` method that:
   - Takes raw audio samples as a numpy array (16kHz, int16)
   - Returns a list of prediction scores between 0-1 for each window

For an example, see the [open_wake_word.py](./models/open_wake_word.py) implementation.
