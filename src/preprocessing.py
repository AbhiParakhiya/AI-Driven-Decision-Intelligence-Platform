import pandas as pd
import numpy as np
import os

def preprocess_data(input_file, output_file):
    """
    Loads raw data, performs cleaning and feature engineering, and saves processed data.
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Date, Region, Category to ensure correct lag/rolling calculations
    df.sort_values(by=['Region', 'Category', 'Date'], inplace=True)
    
    print("Feature Engineering...")
    
    # Date Features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['IsLocalHoliday'] = 0 # Placeholder if needed, already have Holiday_Flag
    
    # Lag Features (Sales from 1, 7, 30 days ago)
    # We need to group by Region and Category before applying shifts
    grouped = df.groupby(['Region', 'Category'])
    
    df['Sales_Lag_1'] = grouped['Sales'].shift(1)
    df['Sales_Lag_7'] = grouped['Sales'].shift(7)
    df['Sales_Lag_30'] = grouped['Sales'].shift(30)
    
    # Rolling Features (Moving Averages)
    df['Sales_Rolling_Mean_7'] = grouped['Sales'].transform(lambda x: x.rolling(window=7).mean())
    df['Sales_Rolling_Mean_30'] = grouped['Sales'].transform(lambda x: x.rolling(window=30).mean())
    df['Sales_Rolling_Std_7'] = grouped['Sales'].transform(lambda x: x.rolling(window=7).std())
    
    # Handle NaNs created by lags/rolling
    df.fillna(0, inplace=True)
    
    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Processed data shape: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'raw', 'sop_data.csv')
    output_path = os.path.join(base_dir, 'data', 'processed', 'sop_data_processed.csv')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    preprocess_data(input_path, output_path)
