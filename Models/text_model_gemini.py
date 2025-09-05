import pandas as pd
import json
import time
from google.generativeai import GenerativeModel
from pathlib import Path
from random import randint

def run_gemini_annotation_pipeline(
    file_paths,
    api_key,
    model_name="gemini-pro",
    save_every=10,
    prompt_col_prefix="prompt_",
):
  
    gen_model = GenerativeModel(model_name)

    def get_response(prompt):
        try:
            response = gen_model.generate_content([{"text": prompt}])
            response_text = response.text.strip().lower()

            if 'yes' in response_text:
                return 1
            elif 'no' in response_text:
                return 0
            else:
                return None
        except Exception as e:
            return str(e)

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        try:
            df = pd.read_pickle(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            continue

        save_path = file_path.with_name(
            file_path.name.replace("Temporary", "Processed").replace(".pkl", "_gemini_text_result.pkl")
        )

        for col in df.columns:
            if prompt_col_prefix in col:
                result_col = f"{col}_result"
                if result_col not in df.columns:
                    df[result_col] = None

                for i in range(len(df)):
                    if pd.isnull(df.at[i, result_col]):
                        try:
                            df.at[i, result_col] = get_response(df.at[i, col])
                        except Exception as e:
                            df.at[i, result_col] = str(e)

                    if i % save_every == 0 and i > 0:
                        result_cols = [c for c in df.columns if '_result' in c]
                        progress_cols = ['idx', 'image_path', 'annotations'] + result_cols
                        if all(col in df.columns for col in progress_cols):
                            df[progress_cols].to_pickle(save_path)
                            print(f"[{file_path.name}] Progress saved at row {i} to {save_path}")
                        else:
                            print(f"Warning: Not all progress columns found in DataFrame for saving at row {i} in {file_path.name}")

        result_cols = [c for c in df.columns if '_result' in c]
        final_cols = ['idx', 'image_path', 'annotations'] + result_cols
        if all(col in df.columns for col in final_cols):
            df[final_cols].to_pickle(save_path)
            print(f"[{file_path.name}] Final file saved to: {save_path}")
        else:
            print(f"Warning: Not all final columns found in DataFrame for saving in {file_path.name}")