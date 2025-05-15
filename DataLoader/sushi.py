import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path

with open("bedtime/parameter.json", "r") as f:
    params = json.load(f)

# Extract the base directory
base_dir = params.get("base_directory", ".")# Default to current directory if missing
results_dir = "Results"
# Change working directory
os.chdir(base_dir)

def read_series(file_path):
        try:
            series_path = os.path.join(base_dir, "DataSet", "Sushi", file_path)
            series_data = pd.read_csv(series_path, header=None).values.flatten()[:2048]  # Assuming a max length of 2048
            return series_data.tolist()  # Convert to list for storage
        except Exception as e:
            return None  # Return None if there's an error (e.g., file missing)

def get_sushi_data(results_dir):
    df = pd.read_csv("DataSet/Sushi/generated_files_list.csv")
    df["series_2048"] = df["File path"].apply(read_series)
    df["series_2048"] = df["series_2048"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # Create 'id' column by stripping '.csv' from 'File path'
    df["idx"] = df["File path"].str.replace(".csv", "", regex=False) 
    df["image_path"] = df["File path"].str.replace(".csv", ".png", regex=False)
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(base_dir, "DataSet", "Sushi", x))
    df[["class", "subclass"]] = df["Class"].str.split(";", expand=True).apply(lambda x: x.str.strip())
    # Function to read series data from CSV files
    df = df.drop(columns=["File path","Class"])
    df.rename(columns={"Caption": "annotations"}, inplace=True)
    df['annotations'] = df['annotations'].apply(lambda x: x.strip().lower())
    df = df[["idx", "series_2048", "annotations", "class", "subclass", "image_path"]]
    df.to_pickle(os.path.join(results_dir, "base_sushi.pkl"))
    return df

df1 = get_sushi_data(results_dir)