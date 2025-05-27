"""
Customer Churn Predictor
Author: tihassfjord
GitHub: github.com/tihassfjord
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("tihassfjord: Predicting which customers are ready to bail.")
    
    # Load the data
    df = pd.read_csv('data/churn.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    
    # Feature engineering
    print("\ntihassfjord: Engineering features to catch those fleeing customers...")
    
    # Encode categorical variables
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    
    # Handle target variable
    target = 'Churn'
    df[target] = df[target].map({'Yes': 1, 'No': 0})
    
    # Select features (use what's available in the dataset)
    numeric_features = ['tenure', 'MonthlyCharges', 'gender']
    available_features = [f for f in numeric_features if f in df.columns]
    
    if not available_features:
        # If standard features aren't available, use numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        available_features = [col for col in numeric_cols if col != target][:3]
    
    print(f"Using features: {available_features}")
    
    X = df[available_features]
    y = df[target]
    
    print(f"Churn rate: {y.mean():.2%}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train the model
    print("\ntihassfjord: Training the churn detector...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_proba)
    accuracy = model.score(X_test, y_test)
    
    print(f"\ntihassfjord's Churn Model Results:")
    print(f"ROC AUC: {auc:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importances:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title("Feature Importance - tihassfjord's Churn Predictor")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
    plt.title("Confusion Matrix - tihassfjord")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))
    
    # High-risk customers
    high_risk_customers = X_test[y_proba > 0.7]
    print(f"\nHigh-risk customers (>70% churn probability): {len(high_risk_customers)}")

if __name__ == "__main__":
    main()
