import pandas as pd
import json
import time
from openai import AzureOpenAI
from random import randint

def run_gpt4_annotation_pipeline(
    file_paths,
    api_key,
    azure_endpoint,
    model_name="gpt-4o",
    api_version="2023-07-01-preview",
    save_every=10,
    prompt_col_prefix="prompt_",
):

    client = AzureOpenAI(
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
    )

    def get_response(prompt):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        response = json.loads(completion.model_dump_json())
        message = response["choices"][0]["message"]["content"].strip().lower()

        if 'yes' in message:
            return 1
        elif 'no' in message:
            return 0
        else:
            return None

    for file_path in file_paths:
        df = pd.read_pickle(file_path)
        save_path = file_path.replace("Temporary", "Processed").replace(".pkl", "_gpt4_text_result.pkl")

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
                        progress_cols = ['idx', 'image_path', 'annotations'] + [c for c in df.columns if '_result' in c]
                        df[progress_cols].to_pickle(save_path)
                        print(f"[{file_path}] Progress saved at row {i}")

        final_cols = ['idx', 'image_path', 'annotations'] + [c for c in df.columns if '_result' in c]
        df[final_cols].to_pickle(save_path)
        print(f"[{file_path}] Final file saved to: {save_path}")
