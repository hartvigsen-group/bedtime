import pandas as pd
import json
from random import randint
import transformers
import torch

def run_llama_annotation_pipeline(
    file_paths,
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    hf_token="hf_YOUR_TOKEN_HERE",
    save_every=10,
    prompt_col_prefix="prompt_",
    save_suffix="_llama_text_result.pkl"
):

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=hf_token,
    )

    def get_response(prompt):
        messages = [{"role": "user", "content": prompt}]
        outputs = pipeline(messages, max_new_tokens=512)
        try:
            return outputs[0]["generated_text"][1]["content"].strip().lower()
        except Exception as e:
            print(f"Error parsing response: {e}")
            return str(e)

    for file_path in file_paths:
        df = pd.read_pickle(file_path)
        save_path = file_path.replace("Temporary", "Processed").replace(".pkl", save_suffix)

        for col in df.columns:
            if prompt_col_prefix in col:
                result_col = f"{col}_result"
                if result_col not in df.columns:
                    df[result_col] = None

                for i in range(len(df)):
                    if pd.isnull(df.at[i, result_col]):
                        try:
                            df.at[i, result_col] = get_response(df[col][i])
                        except Exception as e:
                            df.at[i, result_col] = str(e)

                    if i % save_every == 0 and i > 0:
                        progress_cols = ['idx', 'image_path', 'annotations'] + [c for c in df.columns if '_result' in c]
                        df[progress_cols].to_pickle(save_path)
                        print(f"[{file_path}] Progress saved at row {i}")

        final_cols = ['idx', 'image_path', 'annotations'] + [c for c in df.columns if '_result' in c]
        df[final_cols].to_pickle(save_path)
        print(f"[{file_path}] Final file saved to: {save_path}")
