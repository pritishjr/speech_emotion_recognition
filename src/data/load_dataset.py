'''
we want to lead an audio file and a video file for every sample.

'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class MultimodalData(Dataset):
    
    def __init__(self, audio_dir, video_dir):
        super(MultimodalData, self).__init__()
        
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        
        