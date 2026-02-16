#script to run raw data to processed data
#converting AO, VO, AV formatted files to metadata.csv (interim)

#metadata is for both the modalitities.

import argparse
import pandas as pd
from pathlib import Path

#extracting metadata from the file names in raw.
def parsing_labels(sample_path):
    
    #mapping emotions into a dict/hashmap
    emotion_mappings = {
        '01':'neutral',
        '02':'calm',
        '03':'happy',
        '04':'sad',
        '05':'angry',
        '06':'fearful',
        '07':'disgust',
        '08':'surprised'
    }
    
    #getting the file name without extention:
    sample_name = sample_path.stem
    #splitting it from '-'
    num_parts = sample_name.split('-') #num_parts is returned as a list
    
    if len(num_parts) != 7:
        raise ValueError(f"Invalid file format. Check the path once again.")
    
    #extracting the actor number from the 
    actor_id = int(num_parts[6])
    
    #extracting modality:
    # modality_label = ""
    # if num_parts[0] == 1:
    #     modality_label = 'Audio-Visual (AV)'
    # elif num_parts[0] == 2:
    #     modality_label = 'Video-Only'
    # else:
    #     modality_label = 'Audio-Only'
    
    #extracting the final labels from the sample name:
    label_parts = {
        "Modality":  "Audio" if num_parts[0] == '03' else "Video",
        #voice channel taken for this project is purely SPEECH. no SONG.
        "Emotion" : emotion_mappings,
        "Emotional Intensity" : 'Normal' if num_parts[3] == '01' else 'Strong',
        "Statements" : "Kids are talking by the door" if num_parts[4] == '01' else "Dogs are sitting by the door",
        #"Repetition" : (is it even reqd?)
        "Actor" : actor_id,
        "Sample_path": str(sample_path)
    }
    
    return label_parts

def main(input_dir, output_dir): 
    
    raw_path = Path(input_dir)
    
    #in terms of .csv 
    interim_data = [] 
    
    print("Crawling for audio and video files...")
    
    #crawling and parsing the metadata.
    for ext in ["*.wav, *.mp4"]: #searches these filetypes in the repo
        for sample in raw_path.rglob(ext): 
            #searches the raw path dir in all the directories
            
            meta = parsing_labels(sample)
            if meta:
                interim_data.append(meta)
                
    df = pd.DataFrame(interim_data) #converting to a filetype
    
    #exporting the data (in csv form):
    df.to_csv(output_dir, index=False) #serial numbers are excluded.
    #output directory for us is - /data/interim/interim_data.csv
    
if __name__ == "__main__":
    #creating a parser.
    parser = argparse.ArgumentParser(description= "This program exports the interim metadata from the input directory to the ouptut directory.")
    
    #creating parser CLI arguments:
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/raw", 
        help="Path to input directory."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="/data/interim/interim_data.csv", 
        help="Path to the output directory where you want it to be."
    )
    
    #parsing:
    args = parser.parse_args()
    
    main(args.input, args.output)
    
    
    