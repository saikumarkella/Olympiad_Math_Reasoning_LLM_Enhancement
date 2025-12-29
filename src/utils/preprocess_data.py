"""
    Preprocessing data utilities.
"""
import pandas as pd
import json
import numpy as np
import re
import yaml



def load_json(file_path):
    """Load a JSON file and return its content as a dictionary."""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def merge_jsons_convert_dataframe(json_files):
    """Merge multiple JSON files and convert to a pandas DataFrame."""
    merged_data = []
    for file_path in json_files:
        data = load_json(file_path)
        merged_data.extend(data)
    df = pd.DataFrame(merged_data)
    return df

def remove_invalid_features(df, invalid_features):
    """Remove invalid features from the DataFrame."""
    return df.drop(columns=invalid_features, errors='ignore')



# sanity checking
if __name__ == "__main__":
    # load the yaml config file
    with open("config/main.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # getting the datafiles
    datafiles = config['data']['datafiles']
    print("Total number of datafiles contains : ", len(datafiles))

    df = merge_jsons_convert_dataframe(datafiles)
    print("Dataframe shape after merging JSONs: ", df.shape)
    print(df.head())



