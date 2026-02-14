
import torch
import pandas as pd

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import librosa
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from pathlib import Path

#path handling using pathlib: make paths absolute relative to this script
repo_root = Path(__file__).resolve().parents[2]

raw_audio_dir = repo_root / "data/raw/RAVDESS-speech-audio"
raw_video_dir = repo_root / "data/raw/RAVDESS-speech-video"
speech_audio_files = list(raw_audio_dir.glob("*.wav"))

#Example of an audio file in the RAVDESS dataset (resolved absolute path):
example_audio_path = raw_audio_dir / "Actor_03" / "03-01-03-02-02-02-03.wav"

#current working directory:
# print(f"cwd: {Path.cwd()}")
# print(f"resolved example_audio_path: {example_audio_path}")

if example_audio_path.exists():
    #loading the sample: tuple (signal array, native sampling rate)
    array, sampling_rate = librosa.load(example_audio_path, sr=None)
    
    #using waveshow to visualize the signal:
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(array, sr=sampling_rate)
    plt.show()
    #does not show because the gui-backend is 'headless' and not 'Agg' and there
    
    #saving the file in our repo:
    # plt.savefig("waveform_example_output.png", dpi=300)
    # plt.close()
    
else:
    print("Audio file not found.")
    

