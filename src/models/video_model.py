
'''
input shape:    (batch, vid_frames, channels, height, width) #channel first!
output shape:   sequence/timeline: (batch, seq length, 768)
                pooled: (batch, 768)
                
'''
from huggingface_hub import login
from config import hf_token
if hf_token:
    login(token = hf_token)

import torch
import torch.nn as nn
from transformers import VivitModel
from transformers import logging

class VideoFeatureExtractor(nn.Module):
    
    def __init__(self, pretrained_model='google/vivit-b-16x2-kinetics400', freeze_layers = 0):
        super(VideoFeatureExtractor, self).__init__()
        
        #initializing the vivit model
        self.vivit = VivitModel.from_pretrained(pretrained_model, use_safetensors=True)
        
        #embeddings chop up the frames into tubeletes(3D)
        for param in self.vivit.embeddings.parameters():
            param.requires_grad=False #no updating/backprop the params
                
        if freeze_layers>0:
            #freezing the last 8 layers' params
            for param in self.vivit.encoder.layer[:freeze_layers].parameters():
                param.requires_grad = False
        
    def forward(self, vals):
        
        #we have two outputs given for the input.
        outputs = self.vivit(vals)
        
        #sequence/timeline: the last hidden captures this, important for the cross attention fusion of modalities.
        sequence_output = outputs.last_hidden_state
        
        #pooled
        pooled_output = outputs.pooler_output
        
        return sequence_output, pooled_output
    
def main():

    #taking only one batch, rest the same.
    # example = torch.randn(1, 32, 3, 224, 224)
    
    # #loading the vivit pretrained model:
    # logging.set_verbosity_info()
    # print("Loading the ViViT model.")
    # model = VideoFeatureExtractor(freeze_layers=8)
    
    # print("giving output for the example.")
    # seq_out , pool_out = model(example)
    
    # print(f"""
    #       Checking the shapes:
    #       input tensor shape (4D) : {example.shape}
    #       seq_out shape: {seq_out.shape}
    #       pool_out shape: {pool_out.shape}
    #       """)
    
    pass

if __name__ == "__main__":
    main()