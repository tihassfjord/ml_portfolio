"""
tihassfjord's Sentiment Analysis with IMDB Reviews
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

def decode_review(text, word_index):
    """Decode numerical review back to text"""
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def create_simple_model(vocab_size, max_length):
    """Create simple embedding + global average pooling model"""
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, 16, input_length=max_length),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_lstm_model(vocab_size, max_length):
    """Create LSTM-based model"""
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, 64, input_length=max_length),
        layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
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

def test_on_custom_reviews(model, word_index, max_length):
    """Test model on custom reviews"""
    print("\n🧪 Testing on custom reviews:")
    
    custom_reviews = [
        "This movie was absolutely fantastic! Amazing acting and great story.",
        "Terrible movie. Waste of time. Very boring and poorly made.",
        "It was okay, not great but not terrible either.",
        "Best film I've ever seen! Incredible cinematography and soundtrack.",
        "Completely awful. Bad acting, terrible plot, just horrible."
    ]
    
    for review in custom_reviews:
        # Simple preprocessing (this is basic - in practice you'd want better tokenization)
        words = review.lower().split()
        sequence = []
        for word in words:
            if word in word_index:
                sequence.append(word_index[word])
            # Skip unknown words
        
        # Pad sequence
        if len(sequence) > 0:
            padded = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
            prediction = model.predict(padded, verbose=0)[0][0]
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            print(f"Review: \"{review[:60]}{'...' if len(review) > 60 else ''}\"")
            print(f"Prediction: {sentiment} (confidence: {prediction:.3f})")
            print()

def main():
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow is required for this project.")
        print("Install it with: pip install tensorflow")
        return
    
    print("📝 tihassfjord: Training a text sentiment model.")
    
    # Load IMDB dataset
    print("Loading IMDB dataset...")
    vocab_size = 10000
    max_length = 200
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=vocab_size
    )
    
    print(f"Training sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Get word index for later use
    word_index = tf.keras.datasets.imdb.get_word_index()
    
    # Show some sample data
    print("\nSample review lengths:")
    sample_lengths = [len(x) for x in X_train[:10]]
    print(f"First 10 review lengths: {sample_lengths}")
    print(f"Average length: {np.mean([len(x) for x in X_train]):.1f}")
    
    # Show a sample review
    print(f"\nSample review (first 5 words): {X_train[0][:5]}")
    print(f"Sample decoded: {decode_review(X_train[0], word_index)[:100]}...")
    print(f"Label: {'Positive' if y_train[0] == 1 else 'Negative'}")
    
    # Pad sequences
    print(f"\nPadding sequences to length {max_length}...")
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Ask user for model choice
    print("\nChoose model architecture:")
    print("1. Simple Embedding + Global Average Pooling")
    print("2. LSTM-based model")
    
    choice = input("Enter choice (1 or 2) [default: 1]: ").strip()
    if choice == "2":
        print("Creating LSTM model...")
        model = create_lstm_model(vocab_size, max_length)
        model_name = "LSTM"
        epochs = 3
    else:
        print("Creating simple embedding model...")
        model = create_simple_model(vocab_size, max_length)
        model_name = "Embedding"
        epochs = 5
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print(f"\n{model_name} Model Summary:")
    model.summary()
    
    # Train model
    print(f"\nTraining {model_name} model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=512,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n🎯 tihassfjord's IMDB Sentiment Results:")
    print(f"Model: {model_name}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Test on custom reviews
    test_on_custom_reviews(model, word_index, max_length)
    
    # Analyze predictions
    print("📊 Analyzing test predictions...")
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    
    # Calculate some statistics
    true_positives = np.sum((y_test == 1) & (predicted_labels == 1))
    true_negatives = np.sum((y_test == 0) & (predicted_labels == 0))
    false_positives = np.sum((y_test == 0) & (predicted_labels == 1))
    false_negatives = np.sum((y_test == 1) & (predicted_labels == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    
    # Save model
    model_filename = f"sentiment_{model_name.lower()}_tihassfjord.h5"
    model.save(model_filename)
    print(f"\n💾 Model saved as: {model_filename}")
    
    # Show some prediction examples
    print("\n🔍 Sample predictions:")
    for i in range(5):
        actual = "Positive" if y_test[i] == 1 else "Negative"
        predicted = "Positive" if predicted_labels[i] == 1 else "Negative"
        confidence = predictions[i][0]
        print(f"Review {i+1}: Actual={actual}, Predicted={predicted} (conf: {confidence:.3f})")

if __name__ == "__main__":
    main()
