
'''
input:  1-dimensional vector - raw audio waveform
        shape: (amplitude or intensity) 
output: 768-dimensional vector embedding for every time-step.
        shape: (batch, time_steps, 768)
        
        we need to compress this timeline into a single summary vector.
        therefore, we use mean pooling. but we cannot use torch.mean() since our .npy samples are not all of the same lenghth. hence, we apply a padding vector (attention_mask) to the last hidden layer of hubert.
'''

import torch
import torch.nn as nn
from transformers import logging
#see documentation:
from transformers import HubertModel

class HuBERTFeatureExtractor(nn.Module):
    
    def __init__(self, pre_trained_model = 'facebook/hubert-base-ls960', freeze_cnn = True, freeze_layers=0):
        super(HuBERTFeatureExtractor, self).__init__()
        
        #loading the model:
        self.hubert = HubertModel.from_pretrained(pre_trained_model)
        
        #freezing the cnn encoder since the model is already pre-trained.
        if freeze_cnn: 
            self.hubert.feature_extractor._freeze_parameters()
            
        #freezing N layers at the bottom to prevent overfitting.
        if freeze_layers>0:
            for param in self.hubert.encoder.layers[:freeze_layers].parameters():
                param.requires_grad = False
            
    def forward(self, x, attention_mask=None):
        #the attenction mask is a padding vector that fills the empty time steps with 0 indicating silence and 1 indicating real audio.
        #we need to do this so hubert gets all the 1d waveforms of the same length.
        #attention_mask shape: (batch, time_steps)
        
        #passing the input .npy files to hubert.
        output = self.hubert(x) #shape (batch, hidden_states, 768)
        
        #extracting the last hidden state.
        last_hidden_state = output.last_hidden_state
        #shape: (batch, time_steps, 768)
        
        if attention_mask is not None:
            # If batching clips of different lengths, we must ignore the padding (zeros) so it doesnt affect the average.
            
            #unsqueeze adds a new dim to the mask, expands the added dim to 768, converts it to float.
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            #mask_expanded shape: (batch, time_steps, 768)
            
            #element-wise mul (applying the mask)
            #axis = 1 (sum over time-steps)
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            #shape: (b,768)
            
            #min prevents NaN/inf
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            # Simple average across time if all clips are the exact same length
            pooled_output = torch.mean(last_hidden_state, dim=1)
            
        return pooled_output # Final Shape: (Batch, 768)    

def main():
    
    #input shape: (batch, 1-dimensional values)
    input_example = torch.randn(1, 64640) #.npy
    
    #load hubert model:
    logging.set_verbosity_info()
    print("Loading the HuBERT Model. This may take a while...")
    model = HuBERTFeatureExtractor(freeze_layers=8)
    
    #passing the example for the output:
    output = model(input_example)
    
    #checking the shape of the inputs and outputs:
    print(f"The shape of the input is: {input_example.shape}")
    print(f"The shape of the output is : {output.shape}")
    
if __name__ == "__main__":
    main()
    
    #results:
    #output: (1 , 768)