import pandas as pd
import numpy as np
import json
import os
import sys
from tslearn.metrics import lcss
import random

# Define the folder path
with open("/home/jwm9fu/Time_Series_Description/parameter.json", "r") as f:
    params = json.load(f)

# Extract the base directory
base_dir = params.get("base_directory", ".")# Default to current directory if missing
# Change working directory
os.chdir(base_dir)
results_dir = "Results"

# save_path_stock = os.path.join(results_dir, "base_truce_stock.pkl")
# folder_path_stock = os.path.join(base_dir, "DataSet", "Truce_Stock","raw")
# from DataLoader.truce_stock import get_stock_data
# df_stock = get_stock_data(folder_path_stock,results_dir,save_path_stock)
# df_sushi = get_sushi_data(results_dir)
# save_path_synthetic = os.path.join(results_dir, "base_truce_synthetic.pkl")
# folder_path_synthetic = os.path.join(base_dir, "DataSet", "Truce_Synthetic","raw")
# from DataLoader.truce_synthetic import get_synthetic_data
# df_synthetic = get_synthetic_data(folder_path_synthetic,results_dir,save_path_synthetic)

# Load dataset
df_sushi = pd.read_pickle("Results/base_sushi.pkl")


def get_top_least_similar_series(exp_df, top_n=3):
    series_list = exp_df['series_2048']
    unique_series = list(map(np.array, series_list))  # Ensure arrays are numpy

    series_dict = {}

    # Create a mapping from series (as bytes) to their class
    series_to_class = {np.array(row['series_2048']).tobytes(): row['class'] for _, row in exp_df.iterrows()}

    for i, series_i in enumerate(unique_series):
        max_distances = []
        class_i = series_to_class[series_i.tobytes()]  # Get the class of the current series

        for j, series_j in enumerate(unique_series):
            if i == j:
                continue  # Skip self-comparison
            
            # Ensure series_j comes from a different class
            if series_to_class[series_j.tobytes()] == class_i:
                continue

            try:
                distance = lcss(series_i, series_j)
                max_distances.append((distance, series_j))
            except ValueError as e:
                print(f"LCSS failed for series {i} and {j}: {e}")

        # Sort by distance and keep top_n least similar
        max_distances = sorted(max_distances, key=lambda x: x[0])[:top_n]
        series_dict[series_i.tobytes()] = [idx for _, idx in max_distances]

    return series_dict

# series_dict = get_top_least_similar_series(df_stock[:10], top_n=3)

def process_multiple_files_with_options_lcss_sim(exp_df, base_dir,results_dir,path, top_n=3):
    
    series_dict = get_top_least_similar_series(exp_df, top_n=3)
    series_dict1 = get_top_least_similar_series(exp_df, top_n=1)
    option_columns = []
    correct_indices = []
    most_wrong = []
    
    for idx, row in exp_df.iterrows():
        # print("idx", idx)
        # print("row", row)
        correct_annotation = row['annotations']
        
        # Safely get least similar indices
        similar_indices = series_dict.get( np.array(row['series_2048']).tobytes(), [])
        similar_indices = [arr.tolist() for arr in similar_indices]
        # print("similar_indices",similar_indices)
        # similar_indices1 = series_dict1.get(np.array(row['series_2048']).tobytes(), [])
        # similar_indices1 = [arr.tolist() for arr in similar_indices1]
        # print("similar_indices1",similar_indices1)
        # Check if there are valid alternatives; fallback if empty
        if similar_indices:
            incorrect_annotations = [ exp_df.loc[exp_df['series_2048'].apply(lambda x: list(x) == list(similar_indices[i])), 'annotations'].values for i in range(len(similar_indices))]
            # incorrect_annotations1 = incorrect_annotations[0]
        else:
            incorrect_annotations = []
        
        # Add more random annotations if not enough incorrect options
        while len(incorrect_annotations) < top_n:
            incorrect_annotation = random.choice(exp_df['annotations'])
            if incorrect_annotation != correct_annotation:
                incorrect_annotations.append(incorrect_annotation)
        
        # Shuffle and save the options
        options = [correct_annotation] + incorrect_annotations[:top_n]
        random.shuffle(options)
        
        # Save to columns
        option_columns.append(options)
        correct_indices.append(options.index(correct_annotation))
        most_wrong.append(incorrect_annotations[0])

    
    # Assign new columns
    exp_df[['option_1', 'option_2', 'option_3', 'option_4']] = pd.DataFrame(option_columns)
    exp_df['label'] = correct_indices
    exp_df["false_annotations"] = most_wrong

    # Map numeric labels to letter labels
    label_mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    exp_df['label_alphabet'] = exp_df['label'].map(label_mapping)

    exp_df.to_pickle(os.path.join(base_dir,results_dir,path))
    
    return exp_df

# df1 = process_multiple_files_with_options_lcss_sim(df_synthetic, base_dir,results_dir,path = "truce_synthetic_lcss_sim.pkl", top_n=3)
df2 = process_multiple_files_with_options_lcss_sim(df_sushi, base_dir, results_dir, path=f"sushi_lcss_sim.pkl", top_n=3)

