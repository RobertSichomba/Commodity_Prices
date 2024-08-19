import pandas as pd

def load_cleaned_data(file_path='../data/processed_data.csv'):
    """
    Load cleaned data from the specified CSV file.

    Parameters:
        file_path (str): Path to the cleaned data file.

    Returns:
        DataFrame: A pandas DataFrame containing the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    return df

def main():
    # Load the cleaned data
    df = load_cleaned_data()
    
    if df is not None:
        # Display the first few rows of the DataFrame
        print("Data loaded successfully.")
        print(df.head())
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()
