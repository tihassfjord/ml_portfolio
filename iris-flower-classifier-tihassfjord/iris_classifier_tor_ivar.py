"""
Tor-Ivar's Iris Flower Classification Script
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Tor-Ivar: Loading Iris data 🚀")
    
    # Load the data
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Feature names: {list(X.columns)}")
    print(f"Target names: {data.target_names}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    print("\nTor-Ivar: Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTor-Ivar's Results:")
    print(f"Training accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")
    
    # Show some predictions
    print(f"\nSample predictions:")
    for i in range(min(10, len(y_test))):
        predicted_name = data.target_names[y_test_pred[i]]
        actual_name = data.target_names[y_test.iloc[i]]
        print(f"Predicted: {predicted_name}, Actual: {actual_name}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title("Feature Importance - Tor-Ivar's Iris Classifier")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=data.target_names))

if __name__ == "__main__":
    main()
