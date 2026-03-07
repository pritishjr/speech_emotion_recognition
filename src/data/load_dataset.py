'''
we want to lead an audio file and a video file for every sample.

'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class MultimodalData(Dataset):
    
    def __init__(self, audio_dir, video_dir):
        super(MultimodalData, self).__init__()
        
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        
        #getting file names:
        self.filenames = [f for f in os.listdir(audio_dir) if f.endswith('.npy')]
        
        #we can also check whether it is there or not:
        self.filenames = [f for f in self.filenames if os.path.exists(os.path.exists(os.path.join(video_dir, f)))]
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        #to get the audio(.npy) file from the data directory
        filename = self.filenames[idx]
        
        #loading the numpy array from the disk
        audio_path = os.path.join(self.audio_dir, filename)
        video_path = os.path.join(self.video_dir, filename)
        
        #loading the file from the path
        audio_np = np.load(audio_path)
        video_np = np.load(video_path)
        
        #converting the audio.npy file to torch tensor (important step)
        audio_tensor = torch.from_numpy(audio_np)
        
        #updating our video data samples' dimensions to fit the model's requirements: (frames, height, width, channels) -> (frames, channels, height, width)
        video_np = np.transpose(video_np, (0,3,1,2))
        video_tensor = torch.from_numpy(video_np).float()
        
        #we need to normalize the pixel values in the tensor
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255
            
        #extracting the labels from the filename:
        parts = filename.split('-') #a list
        emotion_label = int(parts[2]) #converting from string to int
        
        #labels should start from 0 to 7 and not 1 to 8
        emotion_label = torch.tensor(emotion_label-1, dtype=torch.long)
        
        final = {
            'audio': audio_tensor,
            'video': video_tensor,
            'label': emotion_label,
            'filename': filename
        }    
        
        return final
    
#since our audio .npy files are not of the same dims/shape due to different lengths of recordings.
def collate_fn(batch):
    
    # 1. Separate the dictionary pieces
    audios = [item['audio'] for item in batch]
    videos = [item['video'] for item in batch]
    labels = [item['label'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    # 2. Pad the audio sequences with zeros (silence)
    # This automatically finds the longest audio in this specific batch and pads the rest
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)
    
    # 3. Create the Attention Mask for HuBERT
    # 1 means "real audio", 0 means "fake padding zero"
    attention_masks = torch.zeros_like(padded_audios)
    for i, audio in enumerate(audios):
        attention_masks[i, :len(audio)] = 1
        
    # 4. Stack the rest (which are already perfectly sized)
    stacked_videos = torch.stack(videos)
    stacked_labels = torch.stack(labels)
    
    return {
        'audio': padded_audios,
        'attention_mask': attention_masks, # <-- HuBERT needs this!
        'video': stacked_videos,
        'label': stacked_labels,
        'filename': filenames
    }
    
def main():
    
    test_audio_path = "./data/processed/audio"
    test_video_path = "./data/processed/video"
    
    if os.path.exists(test_audio_path) and os.path.exists(test_video_path):
        dataset = MultimodalData(test_audio_path, test_video_path)
        
        #loading the dataset from the extraccted info
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        #sequencing it in a batch
        batch = next(iter(dataloader))
        
        print("\n--- Dataloader Batch Shapes ---")
        print(f"Audio Batch: {batch['audio'].shape}") 
        print(f"Video Batch: {batch['video'].shape}")
        print(f"Labels:      {batch['label']}")
        print(f"Filenames:   {batch['filename']}")
    else:
        print("Please update test directories with real paths to run the test block.")
                
        
if __name__ == "__main__":
    main()
        