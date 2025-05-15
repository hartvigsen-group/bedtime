import pandas as pd 
import numpy as np
import ast
import os
import matplotlib.pyplot as plt

df1 = pd.read_pickle("bedtime/DataSet/jpmorgan/jp_outliers_data.pkl")
df2 = pd.read_pickle("bedtime/DataSet/jpmorgan/jp_seasonality_data.pkl")
df3 = pd.read_pickle("bedtime/DataSet/jpmorgan/jp_stat-properties-fat-tail_data.pkl")
df4 = pd.read_pickle("bedtime/DataSet/jpmorgan/jp_stat-properties-stationary_data.pkl")
df5 = pd.read_pickle("bedtime/DataSet/jpmorgan/jp_structural-break_data.pkl")
df6 = pd.read_pickle("bedtime/DataSet/jpmorgan/jp_trend_data.pkl")
df7 = pd.read_pickle("bedtime/DataSet/jpmorgan/jp_volatility_data.pkl")
df = pd.concat([df1,df2,df3,df4,df5,df6,df7])

base_path = "bedtime/DataSet/jpmorgan/images"
df["image_path"] = df["image_path"].apply(lambda p: os.path.join(base_path, os.path.basename(p)))

# Ensure directory exists
os.makedirs(base_path, exist_ok=True)

for idx, row in df.iterrows():
    series = row['series']  # directly use the list
    plt.figure(figsize=(6, 3))
    plt.plot(series, marker='o')
    plt.title(f"Time Series Plot - {row['idx']}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(row["image_path"])
    plt.close()