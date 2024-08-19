# src/analyze.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def analyze_trend(df, commodity):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[commodity].values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.coef_[0]
    return trend

def plot_correlation_matrix(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Commodities")
    plt.show()

def detect_outliers(df, commodities):
    for commodity in commodities:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[commodity])
        plt.title(f"Boxplot of {commodity}")
        plt.show()

def main():
    df = pd.read_csv('../data/processed_data.csv')
    commodities = ['WTI', 'COTTON', 'NATURAL_GAS', 'ALUMINUM', 'COPPER', 'WHEAT']
    
    # Analyze trends
    for commodity in commodities:
        trend = analyze_trend(df, commodity)
        print(f"Trend for {commodity}: {trend:.2f}")

    # Plot correlation matrix
    plot_correlation_matrix(df)
    
    # Detect outliers
    detect_outliers(df, commodities)

if __name__ == "__main__":
    main()

