"""
tihassfjord's Linear Regression (from scratch!)
"""

import numpy as np
import matplotlib.pyplot as plt

def linear_regression_from_scratch():
    print("tihassfjord: Training a linear regression model the hard way.")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    X = np.random.rand(n_samples, 1) * 10
    true_slope = 2.5
    true_intercept = 5.0
    y = true_slope * X.flatten() + true_intercept + np.random.randn(n_samples) * 2
    
    # Add bias term (intercept)
    X_b = np.c_[np.ones((n_samples, 1)), X]  # Add x0 = 1 to each instance
    
    # Initialize parameters
    theta = np.random.randn(X_b.shape[1])  # Random initialization
    learning_rate = 0.01
    n_iterations = 1000
    
    # Store cost history for plotting
    cost_history = []
    
    print(f"Initial parameters: {theta}")
    print(f"True parameters: [{true_intercept:.2f}, {true_slope:.2f}]")
    print("\nTraining...")
    
    # Gradient descent
    for epoch in range(n_iterations):
        # Forward pass: compute predictions
        predictions = X_b.dot(theta)
        
        # Compute cost (Mean Squared Error)
        cost = np.mean((predictions - y) ** 2)
        cost_history.append(cost)
        
        # Compute gradients
        gradients = (2/n_samples) * X_b.T.dot(predictions - y)
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {cost:.4f}")
    
    print(f"\nFinal coefficients (tihassfjord): {theta}")
    print(f"Final loss: {cost_history[-1]:.4f}")
    
    # Make predictions
    X_new = np.array([[0], [2], [5], [8]])
    X_new_b = np.c_[np.ones((4, 1)), X_new]
    y_predict = X_new_b.dot(theta)
    
    print(f"\nSample predictions:")
    for i, (x_val, y_val) in enumerate(zip(X_new.flatten(), y_predict)):
        print(f"X={x_val:.1f} -> y={y_val:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Data and fitted line
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data points')
    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    X_plot_b = np.c_[np.ones((100, 1)), X_plot]
    y_plot = X_plot_b.dot(theta)
    plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'tihassfjord\'s fit: y={theta[1]:.2f}x + {theta[0]:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression from Scratch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cost function over time
    plt.subplot(1, 2, 2)
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return theta, cost_history

def compare_with_normal_equation():
    """Compare gradient descent with the normal equation (analytical solution)"""
    print("\n" + "="*50)
    print("tihassfjord: Comparing with the normal equation...")
    
    # Same data as before
    np.random.seed(42)
    n_samples = 100
    X = np.random.rand(n_samples, 1) * 10
    y = 2.5 * X.flatten() + 5.0 + np.random.randn(n_samples) * 2
    X_b = np.c_[np.ones((n_samples, 1)), X]
    
    # Normal equation: theta = (X^T * X)^(-1) * X^T * y
    theta_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    print(f"Normal equation result: {theta_normal}")
    print("This should be very close to our gradient descent result!")

def main():
    print("🧮 tihassfjord's Linear Regression Implementation")
    print("Implementing gradient descent from scratch...\n")
    
    # Run gradient descent implementation
    theta_gd, cost_history = linear_regression_from_scratch()
    
    # Compare with analytical solution
    compare_with_normal_equation()
    
    print(f"\n🎉 Linear regression training complete!")
    print(f"Gradient descent converged to: intercept={theta_gd[0]:.3f}, slope={theta_gd[1]:.3f}")

if __name__ == "__main__":
    main()
