import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from datasets import Dataset

login("USE_YOUR_TOKEN")

url_synthetic = "https://huggingface.co/datasets/HartvigsenGroup/BEDTime/resolve/main/truce_synthetic.csv"
url_stock = "https://huggingface.co/datasets/HartvigsenGroup/BEDTime/resolve/main/truce_stock.csv"
url_sushi = "https://huggingface.co/datasets/HartvigsenGroup/BEDTime/resolve/main/sushi.csv"
url_taxosynth = "https://huggingface.co/datasets/HartvigsenGroup/BEDTime/resolve/main/taxosynth.csv"
synthetic = pd.read_csv(url_synthetic).drop(columns=["Unnamed: 0"])                 
stock = pd.read_csv(url_stock).drop(columns=["Unnamed: 0"])                   
sushi  = pd.read_csv(url_sushi).drop(columns=["Unnamed: 0"])                    
taxosynth = pd.read_csv(url_taxosynth).drop(columns=["Unnamed: 0"])                 
base_path = "bedtime/images"
os.makedirs(base_path, exist_ok=True)

for name, df in {
    "synthetic": synthetic,
    "stock": stock,
    "sushi": sushi,
    "taxosynth": taxosynth,
}.items():
    # Add image_path column for each dataset separately
    df["image_path"] = df["idx"].apply(lambda i: os.path.join(base_path, f"{name}_{i}.png"))

    for idx, row in df.iterrows():
        series = eval(row['series']) if isinstance(row['series'], str) else row['series']
        plt.figure(figsize=(6, 3))
        plt.plot(series, marker='o')
        plt.title(f"{name} - idx {row['idx']}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(row["image_path"])
        plt.close()

synthetic.to_pickle("Results/synthetic.pickle", index=False)
stock.to_pickle("Results/stock.pickle", index=False)
sushi.to_pickle("Results/sushi.pickle", index=False)
taxosynth.to_pickle("Results/taxosynth.pickle", index=False)