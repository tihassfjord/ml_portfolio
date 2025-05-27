# Build Your Own Neural Network (highlight) — tihassfjord

## Goal
Implement a simple fully-connected neural network for classification using only NumPy.

## Dataset
- MNIST (or small subset, can use sklearn's digits)

## Requirements
- Python 3.8+
- numpy
- scikit-learn
- matplotlib

## How to Run
```bash
python simple_nn_tihassfjord.py
```

## Example Output
```
Training neural net (tihassfjord style).
Epoch 0, Loss: 2.13
Epoch 5, Loss: 1.42
Epoch 10, Loss: 0.89
Epoch 15, Loss: 0.67
Test Accuracy (tihassfjord): 0.92
```

## Project Structure
```
nn-from-scratch-tihassfjord/
│
├── simple_nn_tihassfjord.py    # Main neural network implementation
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## Key Features
- Pure NumPy implementation (no deep learning frameworks)
- Manual forward and backward propagation
- ReLU activation and softmax output
- Training loop with gradient descent
- Evaluation on test set

## Learning Outcomes
- Understanding of neural network fundamentals
- Implementation of backpropagation algorithm
- Gradient computation and weight updates
- Performance evaluation and metrics

---
*Project by tihassfjord - Advanced ML Portfolio*
