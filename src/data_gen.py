import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(start_date='2023-01-01', periods=730, regions=None, categories=None):
    """
    Generates synthetic S&OP data mimicking enterprise sales.
    """
    if regions is None:
        regions = ['North', 'South', 'East', 'West']
    if categories is None:
        categories = ['Electronics', 'Furniture', 'Clothing', 'Home_Appliances']
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    data = []
    
    for date in dates:
        for region in regions:
            for category in categories:
                # Base demand
                base_demand = np.random.randint(50, 200)
                
                # Seasonality (weekly)
                if date.weekday() >= 5: # Weekend boost
                    base_demand *= 1.2
                
                # Monthly seasonality (simple sine wave approximation)
                month_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)
                base_demand *= month_factor
                
                # Trend (slight upward trend)
                days_passed = (date - datetime.strptime(start_date, '%Y-%m-%d')).days
                trend = 1 + (days_passed * 0.0005) 
                base_demand *= trend
                
                # Promotions (random spikes)
                is_promo = np.random.choice([0, 1], p=[0.9, 0.1])
                if is_promo:
                    base_demand *= 1.5
                
                # Holiday (simple approximation)
                is_holiday = 0
                if date.month == 12 and date.day > 20: # Christmas season
                    base_demand *= 1.8
                    is_holiday = 1
                
                # Inventory levels (random fluctuations)
                inventory = int(base_demand * np.random.uniform(0.5, 1.5))
                
                # Final Sales with some noise
                sales = int(base_demand + np.random.normal(0, 10))
                sales = max(0, sales) # Ensure no negative sales
                
                data.append({
                    'Date': date,
                    'Region': region,
                    'Category': category,
                    'Sales': sales,
                    'Inventory': inventory,
                    'Promo_Flag': is_promo,
                    'Holiday_Flag': is_holiday
                })
                
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    output_file = os.path.join(output_dir, 'sop_data.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Data generated successfully and saved to {output_file}")
    print(f"Total records: {len(df)}")
    print(df.head())
