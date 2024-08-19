# src/advanced_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

def seasonal_decomposition(df, commodity):
    print(f"Performing seasonal decomposition for {commodity}")
    decomposition = sm.tsa.seasonal_decompose(df[commodity].dropna(), model='additive', period=12)
    decomposition.plot()
    plt.suptitle(f"Seasonal Decomposition of {commodity}")
    plt.show()

def moving_average(df, commodity, window=12):
    print(f"Calculating moving average for {commodity}")
    df[f'{commodity}_MA'] = df[commodity].rolling(window=window).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(df['date'], df[commodity], label=f"{commodity} Price")
    plt.plot(df['date'], df[f'{commodity}_MA'], label=f"{commodity} {window}-Month MA", color='red')
    plt.title(f"{commodity} Price and {window}-Month Moving Average")
    plt.legend()
    plt.show()

def forecast_arima(df, commodity, steps=12):
    print(f"Forecasting ARIMA for {commodity}")
    model = ARIMA(df[commodity].dropna(), order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def main():
    try:
        # Load the processed data
        df = pd.read_csv('../data/processed_data.csv')
        print("Data loaded successfully:")
        print(df.head())  # Print the first few rows of the dataframe

        # Check for the presence of required columns
        commodities = ['WTI', 'COTTON', 'NATURAL_GAS', 'ALUMINUM', 'COPPER', 'WHEAT']
        for commodity in commodities:
            if commodity not in df.columns:
                print(f"Warning: {commodity} column is missing from the data.")
                return
        
        # Perform seasonal decomposition
        for commodity in commodities:
            seasonal_decomposition(df, commodity)
        
        # Calculate and plot moving averages
        for commodity in commodities:
            moving_average(df, commodity)
        
        # Forecast future prices using ARIMA
        for commodity in commodities:
            forecast = forecast_arima(df, commodity)
            print(f"Forecast for {commodity} for next 12 months:\n", forecast)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
