#script to run raw data to processed data
#converting AO, VO, AV formatted files to metadata.csv

#metadata is for both the modalitities.

import argparse
import pandas as pd


#extracting metadata from the file names in raw.
def extracting_labels(sample_path):
    
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
    parts = sample_name.split('-') #parts is returned as a list
    
    if len(parts) != 7:
        raise ValueError(f"Invalid file format. Check the path once again.")
    
    
    
    