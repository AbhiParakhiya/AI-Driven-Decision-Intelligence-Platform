import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import os
import pickle
import json

def train_model(input_file, model_output_path, metrics_output_path):
    """
    Trains an XGBoost model for demand forecasting.
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Feature Selection
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 
                'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_30', 
                'Sales_Rolling_Mean_7', 'Sales_Rolling_Mean_30', 
                'Sales_Rolling_Std_7', 'Promo_Flag', 'Holiday_Flag', 'Inventory']
    target = 'Sales'
    
    # Encode Categorical Variables (Region, Category)
    # For simplicity, we'll use One-Hot Encoding or Label Encoding.
    # XGBoost handles categories well, but needs encoding.
    print("Encoding categorical features...")
    df = pd.get_dummies(df, columns=['Region', 'Category'], drop_first=True)
    
    # Update feature list after encoding
    feature_cols = [col for col in df.columns if col not in ['Date', 'Sales']]
    print(f"Features: {feature_cols}")
    
    # Time-based Split
    # Last 20% for testing
    split_date = df['Date'].quantile(0.8)
    train_df = df[df['Date'] < split_date]
    test_df = df[df['Date'] >= split_date]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]
    
    print(f"Training shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # Train Model
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}")
    
    print(f"Saving model to {model_output_path}...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Save Metrics
    metrics = {"RMSE": rmse, "MAPE": mape}
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)
        
    # Save as pickle for simplicity with feature names
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
        
    # Also save column names to ensure consistency during inference
    cols_path = model_output_path.replace('.pkl', '_columns.json')
    with open(cols_path, 'w') as f:
        json.dump(feature_cols, f)
        
    print("Training complete.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'processed', 'sop_data_processed.csv')
    model_path = os.path.join(base_dir, 'models', 'xgb_model.pkl')
    metrics_path = os.path.join(base_dir, 'models', 'metrics.json')
    
    train_model(input_path, model_path, metrics_path)
