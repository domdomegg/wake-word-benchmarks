#!/usr/bin/env python3
import wave
import numpy as np
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
from models import WakeWordModel, all_models

class WakeWordBenchmark:
    def __init__(self, model: WakeWordModel, pos_sample_dir: str = 'alexa', neg_sample_dir: str = 'common_voice', max_samples: int = 100):
        self.model = model
        self.frame_size = 1280  # 80ms at 16kHz
        self.results = self._run_benchmark(
            pos_sample_dir=pos_sample_dir,
            neg_sample_dir=neg_sample_dir,
            max_samples=max_samples
        )
    
    def _get_max_prediction(self, audio_path: Path):
        """Returns maximum prediction value for the file."""
        with wave.open(str(audio_path), mode='rb') as f:
            # Verify sample rate
            if f.getframerate() != 16000:
                raise ValueError(f"Audio must be 16kHz, got {f.getframerate()}Hz")
            
            # Load WAV clip frames
            audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        
        # Process audio with increasing windows
        predictions = self.model.predict_clip(audio)
        return max(predictions)

    def _run_benchmark(self, pos_sample_dir, neg_sample_dir, max_samples):
        """Run benchmark and collect prediction values for ROC curve."""
        pos_samples = list(Path('data', 'positive_samples/' + pos_sample_dir).glob('**/*.wav'))[:max_samples]
        neg_samples = list(Path('data', 'negative_samples/' + neg_sample_dir).glob('**/*.wav'))[:max_samples]
        
        # Collect all prediction values
        pos_predictions = [self._get_max_prediction(f) for f in tqdm(pos_samples, desc='Testing positive samples')]
        neg_predictions = [self._get_max_prediction(f) for f in tqdm(neg_samples, desc='Testing negative samples')]
        
        # Use quantized thresholds
        thresholds = np.arange(0, 1.01, 0.05)  # 0 to 1 in steps of 0.05
        results = []
        
        for threshold in thresholds:
            false_rejects = sum(p <= threshold for p in pos_predictions)
            false_accepts = sum(p > threshold for p in neg_predictions)
            
            results.append({
                'threshold': threshold,
                'false_reject_rate': false_rejects / len(pos_samples),
                'false_accept_rate': false_accepts / len(neg_samples)
            })
        
        return results

def create_plots():
    # Create figures for both PNG and HTML outputs
    fig_png = go.Figure()
    fig_html = go.Figure()
    
    colors = {'OpenWakeWordModel': '#1f77b4', 'MicroWakeWordModel': '#ff7f0e'}
    
    for Model in all_models:
        print('Starting ' + Model.__name__ + ' benchmark...')
        benchmark = WakeWordBenchmark(Model('alexa'), 'alexa')
        
        # Extract metrics for plotting
        frr = [r['false_reject_rate'] for r in benchmark.results]
        far = [r['false_accept_rate'] for r in benchmark.results]
        thresholds = [r['threshold'] for r in benchmark.results]
        
        # Find point with minimum sum of FRR + FAR
        min_idx = min(range(len(frr)), key=lambda i: frr[i] + far[i])
        min_point = {
            'far': far[min_idx],
            'frr': frr[min_idx],
            'threshold': thresholds[min_idx]
        }
        
        # Create hover text for HTML plot
        hover_text = [f'Threshold: {t:.2f}<br>FRR: {fr:.1%}<br>FAR: {fa:.1%}' 
                     for t, fr, fa in zip(thresholds, frr, far)]
        
        for fig, show_all_thresholds in [(fig_html, True), (fig_png, False)]:
            fig.add_trace(go.Scatter(
                x=far,
                y=frr,
                name=Model.__name__,
                mode='lines+markers',
                line=dict(color=colors[Model.__name__]),
                hovertext=hover_text if show_all_thresholds else None,
                hoverinfo='text' if show_all_thresholds else 'none'
            ))
        
        # Print metrics for minimum point
        print(f"\n{Model.__name__} Minimum Point (t={min_point['threshold']:.3f}):")
        print(f"FRR: {min_point['frr']:.3f}, FAR: {min_point['far']:.3f}")
    
    # Configure layout for both plots
    base_layout = dict(
        title='ROC Curves for Wake Word Models',
        xaxis=dict(
            title='False Accept Rate',
            tickformat=',.0%',
            range=[0, 0.25],
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            title='False Reject Rate',
            tickformat=',.0%',
            range=[0, 0.25],
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='white',
        width=900,
        height=600,
        showlegend=True
    )
    
    fig_png.update_layout(**base_layout)
    fig_html.update_layout(**base_layout)
    
    # Save outputs
    fig_png.write_image('roc_curves.png')
    fig_html.write_html('roc_curves.html')

if __name__ == '__main__':
    create_plots()
