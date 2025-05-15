import pandas as pd
import numpy as np 

def scaling(df, path, scale=100):
    # Select columns that start with 'series'
    series_cols = [col for col in df.columns if col.startswith("series")]
    
    # Convert lists to numpy arrays for faster computation
    df[series_cols] = df[series_cols].applymap(lambda x: np.array(x) if isinstance(x, list) else x)
    
    # Perform scaling and rounding
    df[series_cols] = df[series_cols].applymap(lambda x: np.round(x * scale).astype(int) if isinstance(x, np.ndarray) else x)

    df.to_pickle(path)
    
    return df

# df1 = pd.read_pickle("bedtime/Results/truce_synthetic_interpolated_cos_sim.pkl")
# df2 = scaling(df1,scale=100, path="bedtime/Results/truce_synthetic_interpolated_scaled_cos_sim.pkl")
# df3 = pd.read_pickle("bedtime/Results/truce_synthetic_interpolated_lcss_sim.pkl")
# df4 = scaling(df3,scale=100, path="bedtime/Results/truce_synthetic_interpolated_scaled_lcss_sim.pkl")
# df5 = pd.read_pickle("bedtime/Results/truce_synthetic_interpolated_dtw.pkl")
# df6 = scaling(df5,scale=100, path="bedtime/Results/truce_synthetic_interpolated_scaled_dtw.pkl")
# df7 = pd.read_pickle("bedtime/Results/truce_synthetic_interpolated_euclidean.pkl")
# df8 = scaling(df7,scale=100, path="bedtime/Results/truce_synthetic_interpolated_scaled_euclidean.pkl")

