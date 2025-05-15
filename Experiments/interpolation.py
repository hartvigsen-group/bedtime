import pandas as pd 
import numpy as np 

def interpolate_series(series, new_length):
    old_index = np.arange(len(series))
    new_index = np.linspace(0, len(series) - 1, new_length)
    return np.interp(new_index, old_index, series).tolist()


def interpolate_df(exp_df,path): 
    new_lengths = [24, 36, 48, 60, 72, 84, 96, 108, 120]

    for length in new_lengths:
        exp_df[f'series_{length}'] = exp_df['series_12'].apply(lambda x: interpolate_series(x, length))

    exp_df.to_pickle(path)
    return exp_df

df1 = pd.read_pickle("bedtime/Results/truce_stock_cos_sim.pkl")
exp_df1 = interpolate_df(df1,"bedtime/Results/truce_stock_interpolated_cos_sim.pkl")
df2 = pd.read_pickle("bedtime/Results/truce_stock_dtw.pkl")
exp_df2 = interpolate_df(df2,"bedtime/Results/truce_stock_interpolated_dtw.pkl")
df3 = pd.read_pickle("bedtime/Results/truce_stock_lcss_sim.pkl")
exp_df3 = interpolate_df(df3,"bedtime/Results/truce_stock_interpolated_lcss_sim.pkl")
df4 = pd.read_pickle("bedtime/Results/truce_stock_euclidean.pkl")
exp_df4 = interpolate_df(df4,"bedtime/Results/truce_stock_interpolated_euclidean.pkl")
df5 = pd.read_pickle("bedtime/Results/truce_synthetic_cos_sim.pkl")
exp_df5 = interpolate_df(df5,"bedtime/Results/truce_synthetic_interpolated_cos_sim.pkl")
df6 = pd.read_pickle("bedtime/Results/truce_synthetic_lcss_sim.pkl")
exp_df6 = interpolate_df(df6,"bedtime/Results/truce_synthetic_interpolated_lcss_sim.pkl")
df7 = pd.read_pickle("bedtime/Results/truce_synthetic_euclidean.pkl")
exp_df7 = interpolate_df(df7,"bedtime/Results/truce_synthetic_interpolated_euclidean.pkl")
df8 = pd.read_pickle("bedtime/Results/truce_synthetic_dtw.pkl")
exp_df8 = interpolate_df(df8,"bedtime/Results/truce_synthetic_interpolated_dtw.pkl")

