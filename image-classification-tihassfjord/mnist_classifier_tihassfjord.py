"""
tihassfjord's MNIST Classifier
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

def create_simple_cnn():
    """Create a simple CNN for MNIST classification"""
    model = tf.keras.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

def create_simple_mlp():
    """Create a simple Multi-Layer Perceptron"""
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy - tihassfjord')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss - tihassfjord')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_sample_predictions(model, X_test, y_test, num_samples=9):
    """Plot sample predictions"""
    predictions = model.predict(X_test[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(X_test[i], cmap='gray')
        axes[i].set_title(f'True: {y_test[i]}, Pred: {predicted_classes[i]}')
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions - tihassfjord', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow is required for this project.")
        print("Install it with: pip install tensorflow")
        return
    
    print("🖼️ tihassfjord: Starting MNIST training.")
    
    # Load and preprocess data
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Normalize pixel values to [0, 1]
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    # Show some sample images
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_train[i], cmap='gray')
        plt.title(f'Label: {y_train[i]}')
        plt.axis('off')
    plt.suptitle('Sample MNIST Images - tihassfjord', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Ask user for model choice
    print("\nChoose model architecture:")
    print("1. Simple MLP (Multi-Layer Perceptron)")
    print("2. CNN (Convolutional Neural Network)")
    
    choice = input("Enter choice (1 or 2) [default: 1]: ").strip()
    if choice == "2":
        print("Creating CNN model...")
        model = create_simple_cnn()
        model_name = "CNN"
    else:
        print("Creating MLP model...")
        model = create_simple_mlp()
        model_name = "MLP"
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print(f"\n{model_name} Model Summary:")
    model.summary()
    
    # Train model
    print(f"\nTraining {model_name}...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n🎯 tihassfjord's MNIST Results:")
    print(f"Model: {model_name}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Show sample predictions
    plot_sample_predictions(model, X_test, y_test)
    
    # Save model
    model_filename = f"mnist_{model_name.lower()}_tihassfjord.h5"
    model.save(model_filename)
    print(f"\n💾 Model saved as: {model_filename}")
    
    # Additional analysis
    print("\n📊 Additional Analysis:")
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Confusion matrix-style analysis
    from collections import Counter
    correct_by_digit = Counter()
    total_by_digit = Counter()
    
    for true_label, pred_label in zip(y_test, predicted_classes):
        total_by_digit[true_label] += 1
        if true_label == pred_label:
            correct_by_digit[true_label] += 1
    
    print("Per-digit accuracy:")
    for digit in range(10):
        accuracy = correct_by_digit[digit] / total_by_digit[digit] if total_by_digit[digit] > 0 else 0
        print(f"  Digit {digit}: {accuracy:.3f} ({correct_by_digit[digit]}/{total_by_digit[digit]})")

if __name__ == "__main__":
    main()
