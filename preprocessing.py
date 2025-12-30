# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(df):
    # Convert Date column if available
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        print("No 'Date' column found. Skipping datetime conversion.")

    # Forward fill missing values
    df = df.ffill()

    # Fill any remaining NaNs with 0
    df = df.fillna(0)

    return df

