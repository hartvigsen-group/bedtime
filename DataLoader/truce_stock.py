import pandas as pd
import json
import os
# from draw_plots import plot_and_save_time_series



# Define the folder path
with open("bedtime/parameter.json", "r") as f:
    params = json.load(f)

# Extract the base directory
base_dir = params.get("base_directory", ".")# Default to current directory if missing
results_dir = "Results"
# Change working directory
os.chdir(base_dir)
save_path = os.path.join(results_dir, "base_truce_stock.pkl")

folder_path = os.path.join(base_dir, "DataSet", "Truce_Stock","raw")

def process_json_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    
    # Extract desired columns into a list of dictionaries
    records = []
    for key, value in data.items():
        record = {
            'id': value.get('id'),
            'idx': value.get('idx'),
            'annotations': value.get('annotations'),
            'series': value.get('series')
        }
        records.append(record)
    
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(records)
    
    # Replace '/' with '_' in the 'id' column
    df['id'] = df['id'].str.replace('/', '_')
    
    df["image_path"] = df["id"].apply(lambda x: os.path.join(base_dir, "DataSet", "Truce_Stock", "images",  x))
    
    df = df.drop(columns=['id'])

    df.rename(columns={"series": "series_12"}, inplace=True)
    df['series_12'] = df['series_12'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Explode the annotations into individual rows
    exp_df = df.explode('annotations').reset_index(drop=True)
    
    # Convert annotations to lowercase
    exp_df['annotations'] = exp_df['annotations'].apply(lambda x: x.strip().lower())

    # Drop specific unwanted annotations
    unwanted_annotations = {
        '{}', 'webpage displayed in the url.', 'plot definition, a secret plan or scheme purpose',
        'nothing', '(image link broken) pilot16/ba_weekly_1.png',
        'decreases concave up.\rincreases concave down.\rdecreases concave up.\rconstant.',
        'increases and decreases concave up.\rincreases and remains steady.',
        'peaks in the beginnig slightly\rincreased from slightly end',
        'decreased in middle peaks\rdecreased from end peaks slightly',
        'increased in middle path\rincreased after   end'
    }
    exp_df = exp_df[~exp_df['annotations'].isin(unwanted_annotations)]

    df = df[["idx", "series_12", "annotations", "image_path"]]
    
    return exp_df

def get_json_files_from_folder(folder_path):
    """Retrieves all JSON file paths from the specified folder."""
    # print([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')])
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]

def get_stock_data(folder_path,results_dir,save_path):
    """Processes all JSON files in a folder and stacks them into a single DataFrame."""
    json_file_paths = get_json_files_from_folder(folder_path)
    dataframes = [process_json_file(path) for path in json_file_paths]
    df = pd.concat(dataframes, ignore_index=True)
    df.to_pickle(save_path)
    return df


# Stack JSON files into a single DataFrame
df2 = get_stock_data(folder_path,results_dir,save_path)
