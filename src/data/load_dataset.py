'''
we want to lead an audio file and a video file for every sample.
-performing the tramsposing of the video-frames dimensions.
-performing data augmentation.
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# import torchaudio, torchvision
import torchaudio.transforms as T_audio
import torchvision.transforms as T_video
from torchvision.transforms import v2

def audio_augmentation(waveform, sr=16000):
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0) #adding another dimension to match (B, 64640)
        
    #adding bg noise:
    if torch.rand(1).item() > 0.5:
        noise_amp = 0.005 * torch.rand(1).item()
        noise = torch.randn_like(waveform) * noise_amp
        waveform = waveform + noise 
        #not using add_noise since it is very strict with tensors.
        
    #pitch shifting:
    if torch.rand(1).item() > 0.5:
        
        n_steps = torch.randint(-2,3, (1,)).item()
        if n_steps != 0:
            transform = T_audio.PitchShift(
                sample_rate=sr,
                n_steps=n_steps
            )
            waveform = transform(waveform)
    
    return waveform.squeeze(0) #back to being 1D

def video_augmentation():
    
    #since we have used mediapipe to align the face mesh, we will avoid cropping so the alignment is preserved.
    
    #color jittering:
    col_jit = v2.ColorJitter(
        brightness=0.2,
        contrast=0.4,
        saturation=0.5,
        hue=0.05
    )
    
    #greyscale conversion:
    greyscale = v2.RandomGrayscale(p=0.5)

    #gaussian noise:
    gauss = v2.GaussianNoise(sigma=0.1)
    
    #composing all:
    transformed_vid_frames = v2.Compose([
        col_jit, 
        greyscale, 
        gauss,
        v2.RandomHorizontalFlip(p=0.3), 
        v2.Resize((224,224), antialias=True)
    ])

    return transformed_vid_frames    
    
class MultimodalData(Dataset):

    def __init__(self, audio_dir, video_dir, split='train'):
        super(MultimodalData, self).__init__()

        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.split = split

        # getting file names from the label:
        self.filenames = [f for f in os.listdir(audio_dir) if f.endswith('.npy')]

        # keeping only samples where corresponding video exists
        self.filenames = [f for f in self.filenames if os.path.exists(os.path.join(video_dir, f))]
        
        #initializing the video augmentation to apply later:
        if self.split == 'train':
            self.video_transforms = video_augmentation()
        else:
            self.video_transforms = v2.Compose([
                v2.Resize((224, 224), antialias=True)
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # to get the audio(.npy) file from the data directory
        filename = self.filenames[idx]

        # loading the numpy array from the disk
        audio_path = os.path.join(self.audio_dir, filename)
        video_path = os.path.join(self.video_dir, filename)

        # loading the file from the path
        audio_np = np.load(audio_path)
        video_np = np.load(video_path)

        # converting the audio.npy file to torch tensor (important step)
        audio_tensor = torch.from_numpy(audio_np).float()
        if self.split == 'train':
            audio_augmented = audio_augmentation(audio_tensor)

        # updating our video data samples' dimensions to fit the model's requirements:
        # (frames, height, width, channels) -> (frames, channels, height, width)
        video_np = np.transpose(video_np, (0, 3, 1, 2))
        video_tensor = torch.from_numpy(video_np).float()

        # we need to normalize the pixel values in the tensor
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0
            
        #applying our video_augmentation:
        video_augmented = self.video_transforms(video_tensor)

        # extracting the labels from the filename:
        parts = filename.split('-')  # a list
        emotion_label = int(parts[2])  # converting from string to int

        # labels should start from 0 to 7 and not 1 to 8
        emotion_label = torch.tensor(emotion_label - 1, dtype=torch.long)

        final = {
            'audio': audio_augmented,
            'video': video_augmented,
            'label': emotion_label,
            'filename': filename
        }

        return final


# since our audio .npy files are not of the same dims/shape due to different lengths of recordings.
#custom collate function:
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
        'attention_mask': attention_masks, #imp for HuBERT
        'video': stacked_videos,
        'label': stacked_labels,
        'filename': filenames
    }


def main():
    print("--- Running Dataset Sanity Check ---")
    
    # Replace these with your actual local folder paths
    test_audio_dir = './data/processed/audio' 
    test_video_dir = './data/processed/video'
    
    try:
        # 1. Initialize the dataset
        test_dataset = MultimodalData(audio_dir=test_audio_dir, video_dir=test_video_dir, split='train')
        print(f"Successfully loaded {len(test_dataset)} files.")
        
        # 2. Initialize the DataLoader
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        # 3. Pull exactly one batch to check the mathematics
        for batch in test_loader:
            print("\nBatch loaded successfully! Here are the tensor shapes:")
            print(f"Audio Tensor: {batch['audio'].shape}  --> Expects (Batch, Max_Length)")
            print(f"Attention Mask: {batch['attention_mask'].shape} --> Expects (Batch, Max_Length)")
            print(f"Video Tensor: {batch['video'].shape} --> Expects (Batch, 32, 3, 224, 224)")
            print(f"Labels:       {batch['label'].shape} --> Expects (Batch,)")
            break # Stop after the first batch
            
    except Exception as e:
        print(f"\nCRITICAL ERROR in Dataset: {e}")

if __name__ == "__main__":
    main()
    