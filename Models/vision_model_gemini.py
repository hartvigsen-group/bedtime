import pandas as pd
import json
import google.generativeai as genai
from pathlib import Path

def run_gemini_vision_annotation_pipeline(
    file_paths,
    api_key,
    model_name="gemini-pro-vision",
    save_every=10,
    prompt_col="prompt_vision",
    image_col="encoded_image",
    result_col="prompt_vision_result",
    save_suffix="_gemini_vision_result.pkl"
):
    """
    Uses Gemini Pro Vision to process vision-language prompts in DataFrames.

    Args:
        file_paths (list): List of file paths to .pkl datasets.
        api_key (str): Google Gemini API key.
        model_name (str): Model to call (default: gemini-pro-vision).
        save_every (int): Save progress every N rows.
        prompt_col (str): Name of the prompt column (default: 'prompt_vision').
        image_col (str): Name of the column containing base64-encoded image data.
        result_col (str): Name of the result column.
        save_suffix (str): Suffix for output pickle files.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    def get_response(prompt, base64_image):
        try:
            image_data = base64_image.split(',')[1] if ',' in base64_image else base64_image
            image_bytes = base64.b64decode(image_data)

            response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
            response_text = response.text.strip().lower()

            if 'yes' in response_text:
                return 1
            elif 'no' in response_text:
                return 0
            else:
                return None
        except Exception as e:
            return f"Error: {str(e)}"

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        try:
            df = pd.read_pickle(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            continue

        save_path = file_path.with_name(
            file_path.name.replace("Temporary", "Processed").replace(".pkl", save_suffix)
        )

        if result_col not in df.columns:
            df[result_col] = None

        for i in range(len(df)):
            if pd.isnull(df.at[i, result_col]):
                try:
                    prompt = df.at[i, prompt_col]
                    image_data = df.at[i, image_col]
                    df.at[i, result_col] = get_response(prompt, image_data)
                except Exception as e:
                    df.at[i, result_col] = f"Error: {str(e)}"

            if i % save_every == 0 and i > 0:
                progress_cols = ["idx", "annotations", "image_path", result_col]
                df[progress_cols].to_pickle(save_path)
                print(f"[{file_path.name}] Progress saved at row {i} to {save_path}")

        final_cols = ["idx", "annotations", "image_path", result_col]
        df[final_cols].to_pickle(save_path)
        print(f"[{file_path.name}] Final file saved to: {save_path}")

import base64