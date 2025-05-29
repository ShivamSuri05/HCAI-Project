import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error

def handle_missing(df, col, strategy):
    if strategy == 'drop':
        df = df[df[col].notnull()]
    elif strategy == 'mean':
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
    elif strategy == 'median':
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    elif strategy == 'mode':
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])
    elif strategy == 'keep':
        pass  # do nothing
    return df

def round_decimal(value: float):
    decimal_val = round(value, 4)
    return decimal_val

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
        pass  # do nothing
    return df

def evaluate_classification(y_test, y_pred):
    metrics = {
        'accuracy': round_decimal(accuracy_score(y_test, y_pred)),
        'precision_weighted': round_decimal(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': round_decimal(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_score_weighted': round_decimal(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
    }
    return metrics

def evaluate_regression(y_test, y_pred):
    metrics = {
        'r2_score': round_decimal(r2_score(y_test, y_pred)),
        'mae': round_decimal(mean_absolute_error(y_test, y_pred)),
        'mse': round_decimal(mean_squared_error(y_test, y_pred)),
        'rmse': round_decimal(np.sqrt(mean_squared_error(y_test, y_pred)))
    }
    return metrics