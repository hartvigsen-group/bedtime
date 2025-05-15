import pandas as pd 
import numpy as np 
import json 
import os
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
# save_path_synthetic = os.path.join(results_dir, "base_truce_synthetic.pkl")
# folder_path_synthetic = os.path.join(base_dir, "DataSet", "Truce_Synthetic","raw")
# from DataLoader.truce_synthetic import get_synthetic_data
# df_synthetic = get_synthetic_data(folder_path_synthetic,results_dir,save_path_synthetic)

df_stock = pd.read_pickle("Results/base_truce_stock.pkl")
df_synthetic = pd.read_pickle("Results/base_truce_synthetic.pkl")

def get_top_least_similar_series(exp_df, top_n=3):
    series_list = exp_df['series_12']
    unique_series = list(map(np.array, series_list))  # Ensure arrays
    # print("unique_series",unique_series)
    series_dict = {}
    for i, series_i in enumerate(unique_series):
        print("i",i,series_i)
        max_distances = []
        for j, series_j in enumerate(unique_series):
            print("j",j, series_j)
            if i != j:
                try:
                    distance = lcss(series_i, series_j)
                    # print("distance",distance)
                    max_distances.append((distance, series_j))
                    # print("append",max_distances)
                except ValueError as e:
                    print(f"LCSS failed for series {i} and {j}: {e}")
        
        max_distances = sorted(max_distances, key=lambda x: x[0])[:top_n]
        print("max_distances",max_distances)
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
        similar_indices = series_dict.get( np.array(row['series_12']).tobytes(), [])
        similar_indices = [arr.tolist() for arr in similar_indices]
        # print("similar_indices",similar_indices)
        similar_indices1 = series_dict1.get(np.array(row['series_12']).tobytes(), [])
        similar_indices1 = [arr.tolist() for arr in similar_indices1]
        # print("similar_indices1",similar_indices1)
        # Check if there are valid alternatives; fallback if empty
        if similar_indices:
            incorrect_annotations = [exp_df.loc[
                exp_df['series_12'].apply(lambda x: list(x) == list(similar_indices[i])), 'annotations'
            ].sample(n=1, random_state=np.random.randint(0, 1000)).values[0]  # Pick one at random
            for i in range(len(similar_indices))]
            incorrect_annotations1 = [ random.choice( exp_df.loc[ exp_df['series_12'].apply(lambda x: list(x) == list(similar_indices1[i])), 'annotations'].tolist())for i in range(len(similar_indices1))]
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
        most_wrong.append(incorrect_annotations1[0])

    
    # Assign new columns
    exp_df[['option_1', 'option_2', 'option_3', 'option_4']] = pd.DataFrame(option_columns)
    exp_df['label'] = correct_indices
    exp_df["false_annotations"] = most_wrong

    # Map numeric labels to letter labels
    label_mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    exp_df['label_alphabet'] = exp_df['label'].map(label_mapping)

    exp_df.to_pickle(os.path.join(base_dir,results_dir,path))
    
    return exp_df

df1 = process_multiple_files_with_options_lcss_sim(df_synthetic, base_dir,results_dir,path = "truce_synthetic_lcss_sim.pkl", top_n=3)
df2 = process_multiple_files_with_options_lcss_sim(df_stock, base_dir,results_dir,path = "truce_stock_lcss_sim.pkl", top_n=3)

