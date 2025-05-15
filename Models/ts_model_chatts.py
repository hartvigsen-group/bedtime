import pandas as pd
import torch
import numpy as np
import pickle
import sys
import os
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from sklearn.metrics import f1_score
import numpy.core.numeric as _numeric

# Patch for loading older pickles using numpy
sys.modules['numpy._core.numeric'] = _numeric

# Load ChatTS model and components
def load_chatts_model():
    model = AutoModelForCausalLM.from_pretrained(
        "bytedance-research/ChatTS-14B",
        trust_remote_code=True,
        device_map=0,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "bytedance-research/ChatTS-14B",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        "bytedance-research/ChatTS-14B",
        trust_remote_code=True,
        tokenizer=tokenizer
    )
    return model, tokenizer, processor

def load_ts_desc_data(path, dataset="sushi"):
    with open(path, "rb") as file:
        df = pickle.load(file)

    if dataset == "sushi":
        df["chatts_prompt"] = df["prompt_text"].str.replace(r"\[[^\]]*\]", "<ts><ts/>", regex=True)
    else:
        df["chatts_prompt"] = df["prompt_series_12"].str.replace(r"\[[^\]]*\]", "<ts><ts/>", regex=True)
        raw_ts = df["prompt_series_12"].str.extract(r"\[([^\]]+)\]")[0]
        df["ts"] = raw_ts.str.findall(r"\d+").apply(lambda lst: [int(x) for x in lst])

    return df

def evaluate_sample(row, model, processor, tokenizer, dataset="sushi"):
    prompt = row["chatts_prompt"]
    ts = row["series_2048"] if dataset == "sushi" else row["ts"]

    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"
    inputs = processor(
        text=[prompt],
        timeseries=[np.array(ts)],
        padding=True,
        return_tensors="pt"
    ).to("cuda:0")

    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        use_cache=False
    )

    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
    return response.strip()

def run_chatts_annotation_pipeline(
    file_paths,
    dataset_type="sushi"
):
    model, tokenizer, processor = load_chatts_model()
    results = {}

    for file_path in file_paths:
        df = load_ts_desc_data(file_path, dataset=dataset_type)
        predictions = []

        for i in tqdm(range(len(df)), desc=f"Processing {os.path.basename(file_path)}"):
            row = df.iloc[i]
            try:
                pred = evaluate_sample(row, model, processor, tokenizer, dataset=dataset_type)
                predictions.append(pred)
            except Exception as e:
                predictions.append(f"Error: {e}")

        # Post-process predictions to extract first character (for classification)
        preds_first_char = [s[0].upper() if isinstance(s, str) and s else "?" for s in predictions]
        labels = df["label_alphabet"].str.upper().tolist()

        # Evaluation
        acc = sum(p == l for p, l in zip(preds_first_char, labels)) / len(labels)
        f1 = f1_score(labels, preds_first_char, average="weighted")
    return results
