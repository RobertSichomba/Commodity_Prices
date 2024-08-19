import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(true_values, predicted_values):
    """
    Evaluate the forecasted values against the true values.

    Parameters:
        true_values (array-like): The actual values.
        predicted_values (array-like): The forecasted values.

    Returns:
        tuple: A tuple containing MAE, MSE, and RMSE.
    """
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = mse ** 0.5
    return mae, mse, rmse

def main():
    # Load the true and predicted values
    df = pd.read_csv('../data/processed_data.csv')
    
    # Example: Using the last 12 values as true values
    true_values = df['WTI'][-12:].values  # Adjust as needed
    print("True Values:", true_values)  # Debugging statement
    
    # Dummy predictions for demonstration; replace with your actual predictions
    predicted_values = [value + 0.5 for value in true_values]  # Example modification
    print("Predicted Values:", predicted_values)  # Debugging statement

    # Evaluate the forecast
    mae, mse, rmse = evaluate_forecast(true_values, predicted_values)
    
    # Display evaluation results
    print(f"Evaluation Metrics:\n MAE: {mae}\n MSE: {mse}\n RMSE: {rmse}")

if __name__ == "__main__":
    main()
