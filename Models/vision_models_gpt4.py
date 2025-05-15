import pandas as pd
import json
from openai import AzureOpenAI

def run_gpt4o_vision_annotation_pipeline(
    file_paths,
    api_key,
    azure_endpoint,
    model_name="gpt-4o",
    api_version="2023-07-01-preview",
    save_every=10,
    prompt_col="prompt_vision",
    image_col="encoded_image",
    result_col="prompt_vision_result",
    save_suffix="_gpt4_vision_result.pkl"
):
    """
    Uses GPT-4o Vision (via Azure OpenAI) to process vision-language prompts in DataFrames.

    Args:
        file_paths (list): List of file paths to .pkl datasets.
        api_key (str): Azure OpenAI API key.
        azure_endpoint (str): Azure OpenAI endpoint.
        model_name (str): Model to call (default: gpt-4o).
        api_version (str): API version (default: 2023-07-01-preview).
        save_every (int): Save progress every N rows.
        prompt_col (str): Name of the prompt column (default: 'prompt_vision').
        image_col (str): Name of the column containing base64-encoded image URLs.
        result_col (str): Name of the result column.
        save_suffix (str): Suffix for output pickle files.
    """
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )

    def get_response(prompt, encoded_image_url):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": encoded_image_url}},
                    ],
                }
            ],
        )
        response = completion.model_dump_json(indent=2)
        response_dict = json.loads(response)
        content = response_dict["choices"][0]["message"]["content"].strip().lower()

        if 'yes' in content:
            return 1
        elif 'no' in content:
            return 0
        else:
            return None

    for file_path in file_paths:
        df = pd.read_pickle(file_path)
        save_path = file_path.replace("Temporary", "Processed").replace(".pkl", save_suffix)

        if result_col not in df.columns:
            df[result_col] = None

        for i in range(len(df)):
            if pd.isnull(df.at[i, result_col]):
                try:
                    prompt = df.at[i, prompt_col]
                    image_url = df.at[i, image_col]
                    df.at[i, result_col] = get_response(prompt, image_url)
                except Exception as e:
                    df.at[i, result_col] = f"Error: {str(e)}"

            if i % save_every == 0 and i > 0:
                progress_cols = ["idx", "annotations", "image_path", result_col]
                df[progress_cols].to_pickle(save_path)
                print(f"[{file_path}] Progress saved at row {i}")

        final_cols = ["idx", "annotations", "image_path", result_col]
        df[final_cols].to_pickle(save_path)
        print(f"[{file_path}] Final file saved to: {save_path}")
