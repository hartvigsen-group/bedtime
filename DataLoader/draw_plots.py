import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

def plot_and_save_time_series(dataset_path):
    """
    Reads a dataset from the given path, plots each time series, 
    and saves the plots to their respective image paths.

    Parameters:
    dataset_path (str): Path to the dataset file (CSV format).
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Iterate through each row to generate and save plots
    for _, row in df.iterrows():
        series_data = ast.literal_eval(row['series_12'])  
        image_path = row['image_path']  # Path to save the image

        # Ensure the directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Plot the time series
        plt.figure(figsize=(6, 4))
        plt.plot(series_data, marker='o', linestyle='-', color='b')
        plt.title(f"Time Series Plot for idx {row['idx']}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)

        # Save the figure to the specified path
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

    print("Plots have been saved successfully.")


# plot_and_save_time_series("bedtime/Results/base_truce_stock.csv")
# plot_and_save_time_series("bedtime/Results/base_truce_synthetic.csv")

