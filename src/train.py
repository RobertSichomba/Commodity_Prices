import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Example model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib  # To save the model

def load_cleaned_data(file_path='../data/processed_data.csv'):
    """
    Load cleaned data from the specified CSV file.

    Parameters:
        file_path (str): Path to the cleaned data file.

    Returns:
        DataFrame: A pandas DataFrame containing the cleaned data.
    """
    df = pd.read_csv(file_path)
    return df

def train_model(df):
    """
    Train a machine learning model on the cleaned data.

    Parameters:
        df (DataFrame): The cleaned data.

    Returns:
        model: The trained model.
    """
    # Split data into features and target
    X = df.drop(columns=['WTI'])  # Replace 'WTI' with your target variable
    y = df['WTI']  # Target variable

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model (using Linear Regression as an example)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate and display evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5

    print(f"Model Evaluation Metrics:\n MAE: {mae}\n MSE: {mse}\n RMSE: {rmse}")

    # Save the trained model to a file
    joblib.dump(model, '../data/trained_model.pkl')
    print("Model saved to '../data/trained_model.pkl'.")

    return model

def main():
    # Load the cleaned data
    df = load_cleaned_data()

    # Train the model
    model = train_model(df)

if __name__ == "__main__":
    main()
