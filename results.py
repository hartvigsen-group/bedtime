import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import pickle

def compute_results_summary(results_dict, label_extractor=None):
    """
    Computes accuracy and weighted F1 score for all model outputs.

    Args:
        results_dict (dict): Dictionary with the structure:
            {
                "filename_1": {
                    "predictions": [...],  # model outputs (str or int)
                    "labels": [...]        # ground truth labels (str or int) OR
                    "label_path": "original_file.pkl"
                },
                ...
            }
        label_extractor (func, optional): Function to extract the true label
            from the original dataframe (if 'labels' not present).
            Signature: label_extractor(df) -> list[str]

    Returns:
        pd.DataFrame: Summary table with accuracy and F1 score per dataset.
    """
    summary = []

    for name, data in results_dict.items():
        preds = data["predictions"]

        # Load labels either directly or from file
        if "labels" in data:
            labels = data["labels"]
        elif "label_path" in data:
            with open(data["label_path"], "rb") as f:
                df = pickle.load(f)
            if label_extractor:
                labels = label_extractor(df)
            elif "label_alphabet" in df.columns:
                labels = df["label_alphabet"].str.upper().tolist()
            else:
                raise ValueError(f"No label column found in: {name}")
        else:
            raise ValueError(f"Missing labels for {name}")

        # Normalize predictions
        if isinstance(preds[0], str) and len(preds[0]) > 0:
            preds = [p[0].upper() if isinstance(p, str) else "?" for p in preds]

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')

        summary.append({
            "dataset": name,
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "n_samples": len(labels)
        })

    return pd.DataFrame(summary)
