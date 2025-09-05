import numpy as np
import pandas as pd
import re
from ChatTime_Git.model.model import ChatTime

def run_chattime_annotation_pipeline(
    input_path,
    output_path=None,
    model_path="ChengsenWang/ChatTime-1-7B-Chat",
    sample_first_last_n=250,
    series_lengths=[12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
):

 
    print("Loading model...")
    model = ChatTime(model_path=model_path)

    print(f"Loading dataset from {input_path}")
    df = pd.read_pickle(input_path)

    if sample_first_last_n:
        df = pd.concat([df.head(sample_first_last_n), df.tail(sample_first_last_n)])

    # Step 1: Extract [series] and reformulate prompts
    def extract_series(text):
        match = re.search(r'\[(.*?)\]', text)
        if match:
            return np.array([float(x.strip()) for x in match.group(1).split(',')])
        return None

    def extract_question(text):
        return re.sub(r'\[.*?\]', '[series]', text)

    prompt_cols = [col for col in df.columns if col.startswith("prompt_series_")]
    for col in prompt_cols:
        series_col = col.replace("prompt", "series")
        question_col = col.replace("prompt", "question")
        df[series_col] = df[col].apply(extract_series)
        df[question_col] = df[col].apply(extract_question)

    # Step 2: Evaluate each row for all series lengths
    print("Starting model inference...")
    for i, row in df.iterrows():
        print(f"\n--- Row {i} ---\n")
        for length in series_lengths:
            question_col = f"question_series_{length}"
            series_col = f"series_series_{length}"
            response_col = f"response_series_{length}"

            question = row.get(question_col)
            series = row.get(series_col)

            if question is not None and isinstance(series, np.ndarray):
                try:
                    response = model.analyze(question, series)
                    df.at[i, response_col] = response.lower()
                except Exception as e:
                    print(f"Error on row {i}, length {length}: {e}")
                    df.at[i, response_col] = str(e)

    if output_path is None:
        output_path = input_path.replace("Temporary", "Processed").replace(".pkl", "_chattime.pkl")

    print(f"Saving to {output_path}")
    df.to_pickle(output_path)
    print("Done.")
