import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd

def run_phi_annotation_pipeline(
    file_paths,
    model_id="microsoft/Phi-3.5-mini-instruct",
    save_every=10,
    prompt_col_prefix="prompt_",
    exclude_prompt_substring="prompt_vision",
    save_suffix="_phi_mini_text_result.pkl"
):
  
    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    def get_response(prompt):
        messages = [{"role": "user", "content": prompt}]
        generation_args = {
            "max_new_tokens": 512,
            "return_full_text": False,
            "use_cache": True,
            "do_sample": True
        }
        output = pipe(messages, **generation_args)
        response = output[0]["generated_text"].strip().lower()
        print(response)
        return response

    for file_path in file_paths:
        df = pd.read_pickle(file_path)
        df = df.drop(columns=[col for col in df.columns if exclude_prompt_substring in col.lower()], errors='ignore')
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
