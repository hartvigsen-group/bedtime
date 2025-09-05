import pandas as pd 
import numpy as np 
import json 
import os
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

df_sushi = pd.read_pickle("Results/sushi.pkl")
df_taxosynth = pd.read_pickle("Results/taxosynth.pkl")
# df_sushi = pd.read_pickle("/home/jwm9fu/Time_Series_Description/sample.pkl")

# def get_top_least_similar_series(exp_df, top_n=3):
#     series_list = exp_df['series_2048']
#     unique_series = list(map(np.array, series_list))  # Ensure arrays are numpy

#     series_dict = {}

#     # Create a mapping from series (as bytes) to their class
#     series_to_class = {np.array(row['series_2048']).tobytes(): row['class'] for _, row in exp_df.iterrows()}

#     for i, series_i in enumerate(unique_series):
#         max_distances = []
#         class_i = series_to_class[series_i.tobytes()]  # Get the class of the current series

#         for j, series_j in enumerate(unique_series):
#             if i == j:
#                 continue  # Skip self-comparison
            
#             # Ensure series_j comes from a different class
#             if series_to_class[series_j.tobytes()] == class_i:
#                 continue

#             try:
#                 distance = dtw(series_i, series_j)
#                 max_distances.append((distance, series_j))
#             except ValueError as e:
#                 print(f"DTW Distance failed for series {i} and {j}: {e}")

#         # Sort by distance and keep top_n least similar
#         max_distances = sorted(max_distances, key=lambda x: x[0], reverse=True)[:top_n]
#         series_dict[series_i.tobytes()] = [idx for _, idx in max_distances]

#     return series_dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dtaidistance import dtw  # Faster DTW implementation

# Function to compute DTW distance for a single series against all others
def compute_least_similar(i, unique_series, series_to_class, top_n):
    max_distances = []
    series_i = unique_series[i]
    class_i = series_to_class[series_i.tobytes()]  # Get the class of the current series

    for j, series_j in enumerate(unique_series):
        if i == j:
            continue  # Skip self-comparison
        
        # Ensure series_j comes from a different class
        if series_to_class[series_j.tobytes()] == class_i:
            continue

        try:
            distance = dtw.distance_fast(series_i, series_j)  # Compute DTW distance
            max_distances.append((distance, series_j))
        except ValueError as e:
            print(f"DTW Distance failed for series {i} and {j}: {e}")

    # Sort by distance and keep top_n least similar
    max_distances = sorted(max_distances, key=lambda x: x[0], reverse=True)[:top_n]
    return (series_i.tobytes(), [idx for _, idx in max_distances])

# Parallelized function using Joblib
def get_top_least_similar_series(exp_df, top_n=3):
    series_list = exp_df['series_2048']
    unique_series = list(map(np.array, series_list))  # Ensure arrays are numpy

    # Create a mapping from series (as bytes) to their class
    series_to_class = {np.array(row['series_2048']).tobytes(): row['class'] for _, row in exp_df.iterrows()}

    # Use joblib to parallelize the DTW computation
    num_cores = -1  # Uses all available CPU cores
    results = Parallel(n_jobs=num_cores, verbose=1)(
        delayed(compute_least_similar)(i, unique_series, series_to_class, top_n) for i in range(len(unique_series))
    )

    # Convert results to dictionary
    series_dict = dict(results)
    
    return series_dict


# series_dict = get_top_least_similar_series(df_stock[:10], top_n=3)

def process_multiple_files_with_options_dtw(exp_df, base_dir,results_dir,path, top_n=3):
    
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
            incorrect_annotations = [exp_df.loc[
                exp_df['series_2048'].apply(lambda x: list(x) == list(similar_indices[i])), 'annotations'
            ].sample(n=1, random_state=np.random.randint(0, 1000)).values[0]  # Pick one at random
            for i in range(len(similar_indices))]
            # incorrect_annotations1 = [ random.choice( exp_df.loc[ exp_df['series_2048'].apply(lambda x: list(x) == list(similar_indices1[i])), 'annotations'].tolist())for i in range(len(similar_indices1))]
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

    exp_df['prompt_text'] = exp_df.apply(
    lambda row: (
        f"Carefully analyze the given time series description and choose the single best option that most accurately describes the pattern for the time series {row['series']}."
        f" Follow these rules strictly: (1) Read all options before deciding; (2) Only output the chosen option, highlighted as A, B, C, or D; (3) Avoid adding extra text or explanations."
        f"\nOptions:\n"
        f"A: {row['option_1']}\n"
        f"B: {row['option_2']}\n"
        f"C: {row['option_3']}\n"
        f"D: {row['option_4']}"
    ),
    axis=1)

    exp_df['prompt_true'] = exp_df.apply(
    lambda row: f"""You are tasked with verifying if the provided annotation accurately describes the given time series. 
    Please follow these instructions carefully:

    1. Review the annotation: '{row['annotations']}'.
    2. Analyze the time series: {row['series']}.
    3. Determine if the annotation precisely matches the pattern depicted in the time series.

    Respond only with 'Yes' if the annotation accurately describes the time series. 
    Respond only with 'No' if it does not. Avoid providing any additional comments or explanations.
    """, axis=1

    )

    exp_df['prompt_false'] = exp_df.apply(
        lambda row: f"""You are tasked with verifying if the provided annotation accurately describes the given time series. 
        Please follow these instructions carefully:

        1. Review the annotation: '{row['false annotations']}'.
        2. Analyze the time series: {row['series']}.
        3. Determine if the annotation precisely matches the pattern depicted in the time series.

        Respond only with 'Yes' if the annotation accurately describes the time series. 
        Respond only with 'No' if it does not. Avoid providing any additional comments or explanations.
        """, axis=1

        )

    exp_df.to_pickle(os.path.join(base_dir,results_dir,path))
    
    return exp_df

df2 = process_multiple_files_with_options_dtw(df_sushi, base_dir,results_dir,path = "sushi_dtw.pkl", top_n=3)
df2 = process_multiple_files_with_options_dtw(df_taxosynth, base_dir,results_dir,path = "taxosynth_dtw.pkl", top_n=3)
