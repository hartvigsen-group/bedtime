import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

def run_phi_vision_annotation_pipeline(
    file_paths,
    model_id="microsoft/Phi-3.5-vision-instruct",
    save_every=10,
    prompt_col_substring="prompt_",
    save_suffix="_phi_vision3_result.pkl"
):
   

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        num_crops=4
    )

    def get_response(image_path, prompt_vision):
        messages = [{"role": "user", "content": "<|image_1|>" + prompt_vision}]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(prompt, Image.open(image_path), return_tensors="pt").to("cuda:0")
        generation_args = {"max_new_tokens": 512}

        generate_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=False,
            **generation_args
        )

        # Strip input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip().lower()

        print(response)
        return response

    for file_path in file_paths:
        df = pd.read_pickle(file_path)
        save_path = file_path.replace("Temporary", "Processed").replace(".pkl", save_suffix)

        for col in df.columns:
            if prompt_col_substring in col:
                result_col = f"{col}_result"
                if result_col not in df.columns:
                    df[result_col] = None

                for i in range(len(df)):
                    if pd.isnull(df.at[i, result_col]):
                        try:
                            image_path = df.at[i, "image_path"]
                            prompt = df.at[i, col]
                            df.at[i, result_col] = get_response(image_path, prompt)
                        except Exception as e:
                            df.at[i, result_col] = str(e)

                    if i % save_every == 0 and i > 0:
                        progress_cols = ["idx", "annotations", "image_path"] + [c for c in df.columns if '_result' in c]
                        df[progress_cols].to_pickle(save_path)
                        print(f"[{file_path}] Progress saved at row {i}")

        final_cols = ["idx", "annotations", "image_path"] + [c for c in df.columns if '_result' in c]
        df[final_cols].to_pickle(save_path)
        print(f"[{file_path}] Final file saved to: {save_path}")
