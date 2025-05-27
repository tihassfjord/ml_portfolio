"""
Neural Network from scratch by tihassfjord
A complete implementation of a multi-layer perceptron using only NumPy
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

print("Training neural net (tihassfjord style).")

def load_and_preprocess_data():
    """Load and preprocess the digits dataset"""
    digits = load_digits()
    X = digits.data / 16.0  # Normalize to [0,1]
    
    # One-hot encode targets
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(digits.target.reshape(-1, 1))
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

class NeuralNetwork:
    """Simple neural network implementation from scratch"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        
        # Store for visualization
        self.losses = []
        
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward propagation"""
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """Cross-entropy loss"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-7
        return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
    
    def backward(self, X, y_true, y_pred):
        """Backpropagation"""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = y_pred - y_true
        dW2 = self.a1.T @ dz2 / m
        db2 = np.mean(dz2, axis=0)
        
        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0)
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2):
        """Update weights using gradients"""
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
    
    def train(self, X, y, epochs=20, verbose=True):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y, y_pred)
            
            # Update weights
            self.update_weights(dW1, db1, dW2, db2)
            
            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.2f}")
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)
    
    def accuracy(self, y_true, y_pred):
        """Calculate accuracy"""
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    
    def plot_loss(self):
        """Plot training loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title('Training Loss - Neural Network from Scratch (tihassfjord)')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.grid(True)
        plt.show()

def main():
    """Main training and evaluation loop"""
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {y_train.shape[1]}")
    
    # Initialize network
    input_size = X_train.shape[1]
    hidden_size = 32
    output_size = y_train.shape[1]
    
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1)
    
    # Train the network
    print("\nTraining neural network...")
    nn.train(X_train, y_train, epochs=20)
    
    # Evaluate on test set
    test_pred = nn.predict(X_test)
    test_accuracy = nn.accuracy(y_test, test_pred)
    
    print(f"\nTest Accuracy (tihassfjord): {test_accuracy:.2f}")
    
    # Show some predictions
    print(f"\nSample predictions:")
    for i in range(5):
        true_class = np.argmax(y_test[i])
        pred_class = np.argmax(test_pred[i])
        confidence = test_pred[i][pred_class]
        print(f"Sample {i}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.3f}")
    
    # Plot training loss
    nn.plot_loss()
    
    print("\nNeural network training complete! (tihassfjord)")

if __name__ == "__main__":
    main()
