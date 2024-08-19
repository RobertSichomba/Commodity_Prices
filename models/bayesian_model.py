import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def load_cleaned_data(file_path='../data/processed_data.csv'):
    df = pd.read_csv(file_path)
    return df

def train_bayesian_model(df):
    X = df.drop(columns=['WTI'])  # Replace 'WTI' with your target variable
    y = df['WTI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = BayesianRidge()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5

    print(f"Bayesian Model Evaluation Metrics:\n MAE: {mae}\n MSE: {mse}\n RMSE: {rmse}")

    joblib.dump(model, '../data/bayesian_model.pkl')
    print("Bayesian model saved to '../data/bayesian_model.pkl'.")

def main():
    df = load_cleaned_data()
    train_bayesian_model(df)

if __name__ == "__main__":
    main()
