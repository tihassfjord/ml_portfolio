"""
Tor-Ivar's Titanic Survival Predictor
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Tor-Ivar: Ready to see who lives and dies (in the data).")
    
    # Load the data
    df = pd.read_csv('data/titanic.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Feature engineering
    print("\nTor-Ivar: Engineering features like a boss...")
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'Fare']
    X = df[features]
    y = df['Survived']
    
    print(f"Using features: {features}")
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("\nTor-Ivar: Training the survival predictor...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    acc = model.score(X_val, y_val)
    
    print(f"\nTor-Ivar's Model Results:")
    print(f"Validation Accuracy: {acc:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importances:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title("Feature Importance - Tor-Ivar's Titanic Predictor")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Tor-Ivar")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_val, y_pred))

if __name__ == "__main__":
    main()
