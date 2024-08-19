# src/feature_engineering.py

import pandas as pd

def main():
    # Load the cleaned data
    try:
        df = pd.read_csv('../data/processed_data.csv')
        print("Successfully loaded data.")
    except FileNotFoundError:
        print("The cleaned data file '../data/processed_data.csv' was not found.")
        return
    
    # Check if the DataFrame is empty
    if df.empty:
        print("The DataFrame is empty. Please check the cleaned data file.")
        return
    
    # Display the first few rows and columns for verification
    print("Data before feature engineering:")
    print(df.head())
    print("Columns in the DataFrame:", df.columns)
    
    # Create new features
    # Example: price change for WTI
    if 'WTI' in df.columns:
        df['price_change'] = df['WTI'].pct_change()
    else:
        print("Column 'WTI' not found in the DataFrame.")
        return

    # Save the feature-engineered DataFrame
    output_file = '../data/feature_engineered_data.csv'
    df.to_csv(output_file, index=False)
    
    # Output confirmation
    print(f"Feature engineered data saved to '{output_file}'.")
    print("Feature engineered data preview:")
    print(df.head())  # Display the first few rows of the updated DataFrame

if __name__ == "__main__":
    main()
