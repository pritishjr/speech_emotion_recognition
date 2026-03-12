'''
applying bi-directional cross-attention.
takes out tokens from both the audio and video to form a 768+768 = 1536 dimensional vector.
'''

import torch
import torch.nn as nn
import os
import numpy as np

#importing the audio and video models for fusion.
from audio_model import HuBERTFeatureExtractor
from video_model import VideoFeatureExtractor

class FusionModel(nn.Module):
    def __init__(self, num_classes=8):
        super(FusionModel, self).__init__()
        
        self.audio_extractor = HuBERTFeatureExtractor()
        self.video_extractor = VideoFeatureExtractor()
        
        #cross-attention layers
        self.audio_to_video_attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            batch_first=True,
        )
        self.video_to_audio_attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            batch_first=True,
        )
        #concatenated vector = 1536 dim
        self.classifier = nn.Sequential(
            nn.Linear(1536,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, audio_waveform, vid_frames):
        
        # extracting the features from the raw data inputs.
        # HuBERT extractor currently returns pooled audio: (batch, 768),
        # so we add a singleton token dimension for attention: (batch, 1, 768).
        audio_seq = self.audio_extractor(audio_waveform).unsqueeze(1)
        # ViViT extractor returns (sequence_output, pooled_output); we use sequence_output.
        video_seq = self.video_extractor(vid_frames)[0]
        
        #implementing the bi-directional cross-atttention layer.
        #for the video features to be attended from the audio frames
        v2a_autput, _ = self.video_to_audio_attn(
            query = video_seq,
            key = audio_seq,
            value = audio_seq
        )        
        #for the audio features to be attented from the vid frames
        a2v_output, _ = self.audio_to_video_attn(
            query = audio_seq,
            key = video_seq,
            value = video_seq
        )
        
        #pooling:
        v2a_pooled = v2a_autput.mean(dim=1) #(batch, 768)
        a2v_pooled = a2v_output.mean(dim=1) #(batch, 768)
        
        #concatenating the two vectors: (batch, 1536)
        combined_vector = torch.cat((v2a_pooled, a2v_pooled), dim=1)
        
        #final output:
        logits = self.classifier(combined_vector)
        
        return logits
    
def main():
    #test block:
    
    model = FusionModel()
    
    video_dummy = torch.randn(4,32,3,224,224)
    audio_dummy = torch.randn(4,64640)
    
    #predictions
    output = model(audio_dummy, video_dummy)
    
    print(f"predictions shape: {output.shape}")
    
if __name__ == "__main__":
    main()
    
