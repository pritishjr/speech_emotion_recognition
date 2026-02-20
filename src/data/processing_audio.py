#creating processed data to train our model.

import argparse
import librosa
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm #progress bar
from moviepy import VideoFileClip
#processing every audio file form the AV and making 1D waveforms for each.
#HuBERT takes 1D waveforms as input. Sampling Rate: 16k Hz

def extracting_1d(av_files_dir, target_sr=16000):
    
    av_sample = Path(av_files_dir) # a video+audio
    
    #extracting the video:
    av_video = VideoFileClip(str(av_sample))
    
    #ripping the audio out:
    audio_av = av_video.audio
    #convert to 1d array:
    audio_av = audio_av.to_soundarray()
    
    #converting to mono:
    if (len(audio_av)>1):
        audio_av = np.mean(audio_av, axis=1)
        
    #resampling to 16000Hz:
    og_sr = av_video.audio.fps
    audio_av = librosa.resample(y = audio_av, orig_sr=og_sr, target_sr=target_sr)
    
    av_video.close()
    
    return audio_av #1d tensor

def main():
    
    parser = argparse.ArgumentParser(description="Extracting 1D waveform arrays as the training data for HuBERT")
    #creating 3 parsers:
    
    parser.add_argument(
        "--input",
        type=str,
        default= "/data/interim/metadata.csv",
        help = "Metadata CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/processed/audio",
        help="Output directory where the files need to be stored."
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000, #HuBERT default sampling rate
    )
    
    args = parser.parse_args()
    
    #taking the output path
    output_path = Path(args.output)
    #output_path.mkdir(parents=True, exist_ok=True) 
    #checks if i have the output directory.
    
    #taking the metadata
    df = pd.read_csv(args.input)
    
    #filtering out audio files
    # audio_df = df[df['Sample_path'].str.endswith(".wav")].copy()
    
    #filtering out audio files from the av files:
    audio_df = df[df['Sample_path'].astype(str).str.split('/').str[-1].str.match(r'^01-.*\.mp4$')].copy()
    
    # audio_df.head()
    #new processed samples path: 
    processed_paths = []
    
    print(f"Converting the {len(audio_df)} audio files for the processing...")
    
    #writing the progress bar as we 
    for _, row in tqdm(audio_df.iterrows(), total=len(audio_df)):
        raw_file = Path(row['Sample_path'])
        out_name = f"{raw_file.stem}.npy"
        out_path = output_path / out_name
        
        try:
            # Get raw normalized waveform
            waveform = extracting_1d(raw_file, target_sr=args.sr)
            
            # Save as NumPy array (.npy)
            np.save(out_path, waveform)
            processed_paths.append(str(out_path))
            
        except Exception as e:
            print(f"Skipping {raw_file.name} due to error: {e}")
            processed_paths.append(None)
    
    #saving the processed paths to the audio_df as interim data since it is not processed (not .npy files)
    audio_df["audio_processed_path"] = processed_paths
    audio_df.to_csv("data/interim/audio_metadata.csv", index=False)
    
    print(f"Processing Done. Files are saved in: {args.output}")
    
if __name__ == "__main__":
    main()