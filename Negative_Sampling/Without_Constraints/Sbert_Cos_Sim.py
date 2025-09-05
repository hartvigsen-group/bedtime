import pandas as pd 
import numpy as np 
import json
import os
from sentence_transformers import SentenceTransformer, util
import torch
import random
import numpy as np
import ast 
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

df_stock = pd.read_pickle("Results/stock.pkl")
df_synthetic = pd.read_pickle("Results/synthetic.pkl")

# Function to get the top N least similar annotations
def get_top_least_similar_annotations(exp_df, top_n=3):

    # Load SBERT model globally with CUDA support if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    unique_annotations = exp_df['annotations'].unique()
    # print("unique_annotations",unique_annotations)
    
    # Step 1: Encode the unique annotations using SBERT, move to CUDA if available
    annotation_embeddings = model.encode(unique_annotations, convert_to_tensor=True, device=device)
    
    # Step 2: Move embeddings to GPU for cosine similarity calculations if CUDA is available
    annotation_embeddings = annotation_embeddings.to(device)
    
    # Create a dictionary to store least similar annotations
    annotation_dict = {}
    for i, annotation in enumerate(unique_annotations):
        # Compute cosine similarity between the current annotation and all others
        similarities = util.pytorch_cos_sim(annotation_embeddings[i], annotation_embeddings)[0]
        # print("similarities", similarities)
        
        # Get indices of the top N least similar annotations, excluding self-similarity
        least_similar_indices = torch.topk(similarities, k=len(similarities), largest=False).indices[:top_n]
        # print("least_similar_indices",least_similar_indices)
        
        # Retrieve the actual annotations for these indices
        least_similar_annotations = [unique_annotations[idx] for idx in least_similar_indices]
        # print("least_similar_annotations",least_similar_annotations)
        
        # Add the annotation and its least similar counterparts to the dictionary
        annotation_dict[annotation] = least_similar_annotations
    
    return annotation_dict

# annotation_dict = get_top_least_similar_annotations(df_stock[:5],3)


def process_multiple_files_with_options_cos_sim(combined_exp_df,base_dir,results_dir,path, top_n):
    
    # Get the top 3 least similar annotations across all files
    annotation_dict = get_top_least_similar_annotations(combined_exp_df, top_n= top_n)
    most_dissimilar_dict = get_top_least_similar_annotations(combined_exp_df, top_n= top_n)
    
    # Create lists to hold the options and correct answer indices
    option_columns = []
    correct_indices = []
    most_wrong = []

    # Generate the options and determine the correct index for each annotation
    for _, row in combined_exp_df.iterrows():
        correct_annotation = row['annotations']
        incorrect_annotations = annotation_dict.get(correct_annotation, [])
        most_different_annotation = most_dissimilar_dict.get(correct_annotation, [])

        # Ensure there are enough options (if less than 3, duplicate some)
        while len(incorrect_annotations) < 3:
            incorrect_annotations.append(incorrect_annotations[-1])

        # Combine correct and incorrect annotations, then shuffle
        options = [correct_annotation] + incorrect_annotations[:3]
        random.shuffle(options)

        # Save the shuffled options and the correct index
        option_columns.append(options)
        most_wrong.append(most_different_annotation[0])
        correct_indices.append(options.index(correct_annotation))

    # Add the options and correct index columns to the dataframe
    combined_exp_df[['option_1', 'option_2', 'option_3', 'option_4']] = pd.DataFrame(option_columns)
    combined_exp_df['label'] = correct_indices
    combined_exp_df["false_annotations"] = most_wrong

    # Map numeric labels to letter labels
    label_mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    combined_exp_df['label_alphabet'] = combined_exp_df['label'].map(label_mapping)

    combined_exp_df['prompt_text'] = combined_exp_df.apply(
    lambda row: (
        f"Carefully analyze the given time series description and choose the single best option that most accurately describes the pattern for the time series {row['series']}."
        f" Follow these rules strictly: (1) Read all options before deciding; (2) Only output the chosen option, highlighted as A, B, C, or D; (3) Avoid adding extra text or explanations."
        f"\nOptions:\n"
        f"A: {row['option_1']}\n"
        f"B: {row['option_2']}\n"
        f"C: {row['option_3']}\n"
        f"D: {row['option_4']}"
    ),
    axis=1
)
    combined_exp_df['prompt_true'] = combined_exp_df.apply(
    lambda row: f"""You are tasked with verifying if the provided annotation accurately describes the given time series. 
    Please follow these instructions carefully:

    1. Review the annotation: '{row['annotations']}'.
    2. Analyze the time series: {row['series']}.
    3. Determine if the annotation precisely matches the pattern depicted in the time series.

    Respond only with 'Yes' if the annotation accurately describes the time series. 
    Respond only with 'No' if it does not. Avoid providing any additional comments or explanations.
    """, axis=1

    )

    combined_exp_df['prompt_false'] = combined_exp_df.apply(
        lambda row: f"""You are tasked with verifying if the provided annotation accurately describes the given time series. 
        Please follow these instructions carefully:

        1. Review the annotation: '{row['false annotations']}'.
        2. Analyze the time series: {row['series']}.
        3. Determine if the annotation precisely matches the pattern depicted in the time series.

        Respond only with 'Yes' if the annotation accurately describes the time series. 
        Respond only with 'No' if it does not. Avoid providing any additional comments or explanations.
        """, axis=1

        )

    combined_exp_df.to_pickle(os.path.join(base_dir,results_dir,path))

    return combined_exp_df

df1 = process_multiple_files_with_options_cos_sim(df_stock,base_dir,results_dir,path="truce_stock_cos_sim.pkl", top_n=3)
df2 = process_multiple_files_with_options_cos_sim(df_synthetic,base_dir,results_dir,path="truce_synthetic_cos_sim.pkl",top_n=3)