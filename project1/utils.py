import pandas as pd
import numpy as np

def handle_missing(df, col, strategy):
    if strategy == 'remove':
        df = df[df[col].notnull()]
    elif strategy == 'mean':
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
    elif strategy == 'median':
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    elif strategy == 'mode':
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col].fillna(mode_val[0], inplace=True)
    elif strategy == 'keep':
        pass  # doing nothing
    return df

def handle_outliers(df, col, strategy):
    # Detect outliers using IQR method
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    if strategy == 'remove':
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    elif strategy == 'cap':
        df.loc[df[col] < lower_bound, col] = lower_bound
        df.loc[df[col] > upper_bound, col] = upper_bound
    elif strategy == 'keep':
        pass  # doing nothing
    return df
