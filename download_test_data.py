#!/usr/bin/env python3
import os
import urllib.request
import subprocess
from pathlib import Path
import tarfile
import zipfile
from tqdm import tqdm
import shutil
import json
import argparse
from typing import Optional

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(desc, url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc='Downloading ' + desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def convert_mp3_to_wav(input_path, output_path):
    """Convert MP3 file to 16-bit 16kHz WAV format using ffmpeg."""
    subprocess.run([
        'ffmpeg', '-y',  # Overwrite output files without asking
        '-i', str(input_path),  # Input file
        '-acodec', 'pcm_s16le',  # 16-bit PCM
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',  # Mono
        str(output_path)  # Output file
    ], check=True, capture_output=True)

def download_common_voice(max_samples: Optional[int] = None):
    """Download and process Common Voice delta dataset."""
    api_url = "https://commonvoice.mozilla.org/api/v1/bucket/dataset/cv-corpus-19.0-delta-2024-09-13%2Fcv-corpus-19.0-delta-2024-09-13-en.tar.gz"
    output_dir = Path("data", "negative_samples")
    output_dir.mkdir(exist_ok=True)
    
    print("\nFetching Common Voice download URL...")
    req = urllib.request.Request(api_url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        download_url_signed = data['url']
    
    tar_path = output_dir / "common_voice_delta.tar.gz"
    download_url("negative samples", download_url_signed, tar_path)
    
    print("\nExtracting negative samples...")
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=temp_dir)
    
    # Convert MP3 files to WAV
    print("\nConverting MP3 files to 16-bit 16kHz WAV format...")
    wav_dir = output_dir / "common_voice"
    wav_dir.mkdir(exist_ok=True)
    
    mp3_files = list(temp_dir.rglob("*.mp3"))
    if max_samples is not None:
        mp3_files = mp3_files[:max_samples]
        print(f"\nLimiting to first {max_samples} samples...")
    
    for mp3_file in tqdm(mp3_files, desc="Converting files"):
        wav_file = wav_dir / f"{mp3_file.stem}.wav"
        try:
            convert_mp3_to_wav(mp3_file, wav_file)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {mp3_file}: {e}")
            continue
    
    # Clean up
    print("\nCleaning up temporary files...")
    tar_path.unlink()
    shutil.rmtree(temp_dir)

def download_wake_word_samples():
    """Download wake word samples from domdomegg's benchmark repository."""
    url = "https://github.com/domdomegg/picovoice-wake-word-benchmark/archive/master.zip"
    zip_path = Path("wake-word-benchmark.zip")
    repo_dir = Path("picovoice-wake-word-benchmark-master")
    output_dir = Path("data", "positive_samples")
    
    download_url("wake word samples", url, zip_path)
    
    print("\nExtracting wake word samples...")
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall()
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Move wake word sample folders directly
    print("\nMoving wake word samples...")
    wake_words = ["alexa", "computer", "jarvis", "smart_mirror", "snowboy", "view_glass"]
    for keyword in wake_words:
        src_dir = repo_dir / keyword
        dst_dir = output_dir / keyword
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.move(str(src_dir), str(dst_dir))
            print(f"Moved {keyword} samples")
    
    # Clean up
    zip_path.unlink()
    shutil.rmtree(repo_dir)

def main():
    parser = argparse.ArgumentParser(description="Download wake word benchmark test data")
    parser.add_argument("--max-negative-samples", type=int, help="Maximum number of negative samples to keep from CommonVoice")
    args = parser.parse_args()
    
    print("Downloading test data...")
    
    # Download wake word samples
    download_wake_word_samples()
    
    # Download and process Common Voice samples
    download_common_voice(max_samples=args.max_negative_samples)
    
    print("\nDownload complete!")
    print("\nTest data organization:")
    print("- data/positive_samples/: Wake word samples (alexa, computer, jarvis, smart_mirror, snowboy, view_glass)")
    print("- data/negative_samples/common_voice/: Background speech from Common Voice delta dataset")
    print("\nAll audio files are 16-bit 16kHz WAV format")

if __name__ == "__main__":
    main()
