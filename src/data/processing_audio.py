#creating processed data to train our model.

import argparse
import librosa
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm #progress bar

#processing every audio file and making 1D waveforms for each.
#HuBERT takes 1D waveforms as input. Sampling Rate: 16k Hz

def extracting_1d(sample_path, target_sr=16000):
    
    #loading the sample signal: (array,sampling rate)
    signal, _ = librosa.load(sample_path, sr=target_sr)
    
    #trimming the silence at 30dB: captures breath but trims silence.
    #tuple(trimmed signal, index)
    signal_trimmed, _ = librosa.effects.trim(signal, top_db=30)
    
    #normalizing the input signals.
    if np.std(signal_trimmed)>0:
        
        signal_norm = (signal_trimmed - np.mean(signal_trimmed)) / np.std(signal_trimmed)
    else:
        
        signal_norm = signal_trimmed
    
    return signal_norm #1d array of the signal

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
    #output_path.mkdir(parents=True, exist_ok=True) #checks if i have the output directory.
    
    #taking the metadata
    df = pd.read_csv(args.input)
    
    #filtering out audio files.
    audio_df = df[df['Sample_path'].str.endswith(".wav")].copy()
    
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