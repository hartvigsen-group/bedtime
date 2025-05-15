import pandas as pd
import numpy as np

def apply_random_missingness(series, missing_fraction=0.25, random_seed=None):
    """
    Randomly masks out a fraction of the time series by replacing entries with NaN.

    Args:
        series (list or pd.Series): Input time series.
        missing_fraction (float): Fraction of values to mask (between 0 and 1).
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        list: Time series with randomly masked values (as NaN).
    """
    if isinstance(series, list):
        series = pd.Series(series)
    
    if not 0 <= missing_fraction <= 1:
        raise ValueError("missing_fraction must be between 0 and 1.")
    
    np.random.seed(random_seed)
    mask = np.random.rand(len(series)) < missing_fraction
    masked_series = series.copy()
    masked_series[mask] = np.nan
    
    return masked_series.tolist()
