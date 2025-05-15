import pandas as pd
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def run_qwen_vl_annotation_pipeline(
    file_paths,
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    save_every=10,
    prompt_col_substring="prompt_vision",
    save_suffix="_qwen_vision_result.pkl"
):
    """
    Processes datasets using Qwen2.5-VL-7B-Instruct to evaluate image + prompt pairs.

    Args:
        file_paths (list): List of .pkl file paths to process.
        model_id (str): Hugging Face model identifier.
        save_every (int): Save progress every N rows.
        prompt_col_substring (str): Substring to identify prompt columns (e.g., 'prompt_vision').
        save_suffix (str): Suffix to use for the saved output pickle file.
    """

    print("Loading Qwen2.5-VL model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    def get_response(encoded_image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": encoded_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = processor.batch_decode(
            generated_ids_trimmed,
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
                            encoded_img = df.at[i, "encoded_image"]
                            prompt = df.at[i, col]
                            df.at[i, result_col] = get_response(encoded_img, prompt)
                        except Exception as e:
                            df.at[i, result_col] = str(e)

                    if i % save_every == 0 and i > 0:
                        progress_cols = ["idx", "annotations", "image_path"] + [c for c in df.columns if '_result' in c]
                        df[progress_cols].to_pickle(save_path)
                        print(f"[{file_path}] Progress saved at row {i}")

        final_cols = ["idx", "annotations", "image_path"] + [c for c in df.columns if '_result' in c]
        df[final_cols].to_pickle(save_path)
        print(f"[{file_path}] Final file saved to: {save_path}")
