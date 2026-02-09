import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(input_file, output_dir):
    """
    Loads processed data, prints stats, and saves visualizations.
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating statistical summary...")
    desc = df.describe()
    print(desc)
    desc.to_csv(os.path.join(output_dir, 'statistical_summary.csv'))
    
    print("Generating plots...")
    
    # 1. Sales over time by Region
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Sales', hue='Region')
    plt.title('Daily Sales by Region')
    plt.savefig(os.path.join(output_dir, 'sales_by_region.png'))
    plt.close()
    
    # 2. Sales Distribution by Category
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Category', y='Sales')
    plt.title('Sales Distribution by Category')
    plt.savefig(os.path.join(output_dir, 'sales_by_category_boxplot.png'))
    plt.close()
    
    # 3. Impact of Promotions
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Promo_Flag', y='Sales', estimator="mean")
    plt.title('Average Sales: Promo vs No Promo')
    plt.savefig(os.path.join(output_dir, 'promo_impact.png'))
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    print(f"EDA complete. Plots saved to {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'processed', 'sop_data_processed.csv')
    output_dir = os.path.join(base_dir, 'notebooks', 'eda_plots')
    
    perform_eda(input_path, output_dir)
