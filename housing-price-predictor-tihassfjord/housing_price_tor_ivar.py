"""
Tor-Ivar's Housing Price Predictor
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Tor-Ivar: Predicting house prices like a skeptical economist.")
    
    # Load the data
    df = pd.read_csv('data/housing.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"First few rows:\n{df.head()}")
    
    # Feature engineering (using available features in the sample data)
    print("\nTor-Ivar: Engineering features...")
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Use the available features
    available_features = [col for col in ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF'] 
                         if col in df.columns]
    
    if not available_features:
        # If standard features aren't available, use numeric columns (excluding target)
        target_col = 'SalePrice' if 'SalePrice' in df.columns else df.columns[-1]
        available_features = [col for col in numeric_columns if col != target_col][:4]
        
    print(f"Using features: {available_features}")
    
    X = df[available_features]
    y = df['SalePrice'] if 'SalePrice' in df.columns else df.iloc[:, -1]  # Use last column as target
    
    print(f"Target variable stats:\nMean: ${y.mean():.0f}\nStd: ${y.std():.0f}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("\nTor-Ivar: Training gradient boosting regressor...")
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTor-Ivar's Model Results:")
    print(f"RMSE: ${rmse:.0f}")
    print(f"R² Score: {r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop Features:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title("Feature Importance - Tor-Ivar's Housing Price Predictor")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
    
    # Prediction vs Actual plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices - Tor-Ivar")
    plt.tight_layout()
    plt.show()
    
    # Sample predictions
    print(f"\nSample Predictions:")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        print(f"Actual: ${actual:.0f}, Predicted: ${predicted:.0f}, Diff: ${abs(actual-predicted):.0f}")

if __name__ == "__main__":
    main()
